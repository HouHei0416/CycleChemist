import sys
import os
import math
import json
import torch
import psutil
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem, DataStructs, RDLogger
from torch.utils.tensorboard import SummaryWriter

RDLogger.DisableLog('rdApp.*')

# Custom predictor imports - set up paths first to avoid namespace conflicts
_FINETUNE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_FINETUNE_DIR))

# Add the MatGPT model directory to path to avoid 'model' namespace conflict with property predictors
_MATGPT_MODEL_DIR = os.path.join(_FINETUNE_DIR, 'model')
sys.path.insert(0, _MATGPT_MODEL_DIR)
from MatGPT import MatGPT, MatGPTConfig
sys.path.pop(0)

from vocabulary import read_vocabulary
from utils import (
    set_seed, sample_SMILES, likelihood, to_tensor, 
    calc_fingerprints, int_div, is_valid_smiles, freeze_parameters,
    evaluate_smiles_validity
)

# Add paths for moe2_p3 and opvc modules
_MOE2_P3_DIR = os.path.join(_PROJECT_ROOT, 'property_predictors', 'moe2_p3')
_OPVC_DIR = os.path.join(_PROJECT_ROOT, 'property_predictors', 'opvc')
sys.path.insert(0, _MOE2_P3_DIR)
sys.path.insert(0, _OPVC_DIR)

from inference import load_model, predict_pce, predict_homo_lumo
from model import MOE2, P3
from config import PCE_MODEL_PATH, HOMOLUMO_EXP_MODEL_PATH, PREDICT_DATA_PATH
from opv_predictor import OPVPredictor

def parse_args():
    parser = argparse.ArgumentParser(description="CycleChemist MatGPT Fine-tuning")
    parser.add_argument('--run_name', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--n_steps', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--sigma1', type=float, default=100.0)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--memory_size', type=int, default=1000)
    parser.add_argument('--replay', type=int, default=5)
    parser.add_argument('--sim_penalize', action='store_true', default=False)
    parser.add_argument('--sim_thres', type=float, default=0.7)
    parser.add_argument('--prior_path', type=str, required=True)
    # Default paths (relative to project root)
    default_vocab_path = os.path.join(_PROJECT_ROOT, 'data', 'vocab_pubchem_len20-290_no_ions_no_multi_random5M_123.txt')
    default_ckpt_path = os.path.join(_PROJECT_ROOT, 'matgpt', 'ckpt', 'rl_fine_tune')
    default_mem_path = os.path.join(_PROJECT_ROOT, 'matgpt', 'outputs')
    
    parser.add_argument('--vocab_path', type=str, default=default_vocab_path)
    parser.add_argument('--ckpt_save_path', type=str, default=default_ckpt_path)
    parser.add_argument('--mem_save_path', type=str, default=default_mem_path)
    # Default dataset path (relative to project root)
    default_dataset_path = os.path.join(_PROJECT_ROOT, 'data', 'exp_dataset.csv')
    parser.add_argument('--dataset', type=str, default=default_dataset_path)
    parser.add_argument('--use_homo_lumo_loss', action='store_true', default=False)
    parser.add_argument('--kl_coeff', type=float, default=500)
    parser.add_argument('--use_opv_predictor', action='store_true', default=True)
    # Default OPV predictor path (relative to project root)
    default_opv_path = os.path.join(_PROJECT_ROOT, 'property_predictors', 'opvc', 'model_output', 'ovp_random_forest_model.pkl')
    parser.add_argument('--opv_predictor_path', type=str, default=default_opv_path)
    # Default PCE model path (relative to project root)
    default_pce_path = os.path.join(_PROJECT_ROOT, 'property_predictors', 'moe2_p3', 'ckpt', 'p3', 'PCE_model_fold1.pth')
    parser.add_argument('--pce_model_path', type=str, default=default_pce_path)
    parser.add_argument('--target_type', type=str, required=True)
    parser.add_argument('--fixed_smiles', type=str, required=True)
    parser.add_argument('--freeze_layers', type=int, default=8)
    parser.add_argument('--keep_train_modules', nargs='+', default=['head', 'mlp'])
    parser.add_argument('--freeze_modules', nargs='+', default=[])
    parser.add_argument('--use_tensorboard', action='store_true', default=True)
    parser.add_argument('--n_layer', type=int, default=8)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_embd', type=int, default=256)
    parser.add_argument('--block_size', type=int, default=340)
    parser.add_argument('--weight_decay', type=float, default=0.2)
    parser.add_argument('--use_rotary', action='store_true', default=True)
    parser.add_argument('--use_rel_pos_bias', action='store_true', default=True)
    parser.add_argument('--use_gated_mlp', action='store_true', default=True)
    parser.add_argument('--use_diversity_loss', action='store_true', default=True)
    parser.add_argument('--diversity_weight', type=float, default=0.1)
    parser.add_argument('--nucleus_p', type=float, default=0.91)
    parser.add_argument('--use_temperature_annealing', action='store_true', default=True)
    parser.add_argument('--min_temperature', type=float, default=0.5)
    parser.add_argument('--max_temperature', type=float, default=1.4)
    return parser.parse_args()

class CycleChemistTrainer:
    def __init__(self, configs):
        self.configs = configs
        self.device = torch.device(configs["device"])
        self.voc = read_vocabulary(configs["vocab_path"])
        self.batch_size = configs["batch_size"]
        self.n_steps = configs["n_steps"]
        self.learning_rate = configs["learning_rate"]
        self.sigma1 = configs["sigma1"]
        
        # Experience replay
        self.memory = pd.DataFrame(columns=["smiles", "pce_scores", "scores", "seqs", "fps"])
        self.memory_size = configs["memory_size"]
        self.replay = configs["replay"]
        
        # Diversity/Similarity
        self.sim_penalize = configs["sim_penalize"]
        self.sim_thres = configs["sim_thres"]
        self.kl_coeff = configs.get("kl_coeff", 0.0)
        
        # Models - use updated class names and paths
        self.homo_lumo_model = load_model(MOE2, HOMOLUMO_EXP_MODEL_PATH, self.device)
        # Use pce_model_path from configs (full path, can be overridden via command line)
        pce_model_path = configs.get("pce_model_path")
        if pce_model_path is None:
            raise ValueError("pce_model_path must be provided via --pce_model_path argument")
        self.pce_model = load_model(P3, pce_model_path, self.device)
        self.opv_classifier = OPVPredictor(model_path=configs["opv_predictor_path"])

        # PCE de-normalization statistics (match logic in moe2_p3/inference.py)
        try:
            pce_data = pd.read_csv(PREDICT_DATA_PATH)
            if "PCE" in pce_data.columns:
                self.pce_mean = float(pce_data["PCE"].mean())
                self.pce_std = float(pce_data["PCE"].std())
            else:
                # Fallback: no PCE column, keep normalized scale
                self.pce_mean = 0.0
                self.pce_std = 1.0
        except Exception:
            # If stats cannot be loaded, keep normalized scale
            self.pce_mean = 0.0
            self.pce_std = 1.0
        
        # Ensure output directories exist
        os.makedirs(self.configs["mem_save_path"], exist_ok=True)
        
        self.target_type = configs["target_type"]
        self.fixed_smiles = configs["fixed_smiles"]
        
        if self.target_type == "donor":
            self._load_reference_stats('HOMO_D', 'LUMO_D')
        else:
            self._load_reference_stats('HOMO_A', 'LUMO_A')
            
        self.writer = None
        if configs.get("use_tensorboard", False):
            log_dir = os.path.join(_PROJECT_ROOT, "matgpt", "runs", f"{configs['run_name']}")
            self.writer = SummaryWriter(log_dir=log_dir)

    def _load_reference_stats(self, homo_col, lumo_col):
        data = pd.read_csv(self.configs["dataset"])
        target_data = data[[homo_col, lumo_col]].dropna()
        self.homo_mean = target_data[homo_col].mean()
        self.homo_std = target_data[homo_col].std()
        self.lumo_mean = target_data[lumo_col].mean()
        self.lumo_std = target_data[lumo_col].std()

    def _kl_divergence(self, homo_values, lumo_values):
        homo_values, lumo_values = np.array(homo_values), np.array(lumo_values)
        eps = 1e-8
        
        def kl(mu_p, std_p, mu_q, std_q):
            return np.log(std_q / std_p) + (std_p**2 + (mu_p - mu_q)**2) / (2 * std_q**2) - 0.5
            
        kl_h = kl(np.mean(homo_values), np.std(homo_values) + eps, self.homo_mean, self.homo_std + eps)
        kl_l = kl(np.mean(lumo_values), np.std(lumo_values) + eps, self.lumo_mean, self.lumo_std + eps)
        return float(kl_h + kl_l)

    def _calculate_reward(self, pce_scores, opv_probs):
        pce_scores = np.array(pce_scores)
        if self.configs.get("use_opv_predictor", True):
            probs = np.nan_to_num(np.array(opv_probs[0][1]), nan=0.0)
            return np.clip(pce_scores * probs, 0, 31)
        return np.clip(pce_scores, 0, 31)

    def _memory_update(self, smiles, pce_scores, final_scores, seqs):
        final_scores, pce_scores = list(final_scores), list(pce_scores)
        seqs_list = [seqs[i, :].cpu().numpy() for i in range(len(smiles))]
        fps_memory = list(self.memory["fps"])
        
        for i in range(len(smiles)):
            if pce_scores[i] == -100: continue
            
            fp, smiles_i = calc_fingerprints([smiles[i]])
            if not fp: continue
            
            if self.sim_penalize and fps_memory:
                sims = [DataStructs.FingerprintSimilarity(fp[0], DataStructs.CreateFromBinaryString(bytes.fromhex(x))) for x in fps_memory]
                if np.sum(np.array(sims) >= self.sim_thres) > 20:
                    final_scores[i] = 0
            
            if final_scores[i] > 0:
                new_data = pd.DataFrame({
                    "smiles": [smiles_i[0]],
                    "pce_scores": [pce_scores[i]],
                    "scores": [final_scores[i]],
                    "seqs": [seqs_list[i]],
                    "fps": [fp[0].ToBinary().hex()]
                })
                self.memory = pd.concat([self.memory, new_data], ignore_index=True)
                
        self.memory = self.memory.drop_duplicates(subset=["smiles"]).sort_values('scores', ascending=False).reset_index(drop=True)
        if len(self.memory) > self.memory_size:
            self.memory = self.memory.head(self.memory_size)

        if self.replay > 0 and not self.memory.empty:
            s = min(len(self.memory), self.replay)
            experience = self.memory.head(5 * self.replay).sample(s).reset_index(drop=True)
            smiles += list(experience["smiles"])
            pce_scores += list(experience["pce_scores"])
            final_scores += list(experience["scores"])
            for idx in experience.index:
                exp_seq = torch.tensor(experience.loc[idx, "seqs"], dtype=torch.long, device=self.device).view(1, -1)
                seqs = torch.cat((seqs, exp_seq), dim=0)
                
        return smiles, np.array(pce_scores), np.array(final_scores), seqs

    def train(self):
        ckpt_dir = os.path.join(self.configs["ckpt_save_path"], self.configs["run_name"])
        os.makedirs(ckpt_dir, exist_ok=True)
        
        model_config = MatGPTConfig(
            vocab_size=len(self.voc),
            n_layer=self.configs["n_layer"],
            n_head=self.configs["n_head"],
            n_embd=self.configs["n_embd"],
            block_size=self.configs["block_size"],
            use_rotary=self.configs["use_rotary"],
            use_rel_pos_bias=self.configs["use_rel_pos_bias"],
            use_gated_mlp=self.configs["use_gated_mlp"],
            use_diversity_loss=self.configs["use_diversity_loss"],
            diversity_weight=self.configs["diversity_weight"],
            nucleus_p=self.configs["nucleus_p"],
            use_temperature_annealing=self.configs["use_temperature_annealing"],
            min_temperature=self.configs["min_temperature"],
            max_temperature=self.configs["max_temperature"]
        )

        prior = MatGPT(model_config).to(self.device)
        prior.load_state_dict(torch.load(self.configs["prior_path"]), strict=True)
        prior.eval()
        for p in prior.parameters(): p.requires_grad = False

        agent = MatGPT(model_config).to(self.device)
        agent.load_state_dict(torch.load(self.configs["prior_path"]), strict=True)
        if self.configs["freeze_layers"] > 0:
            freeze_parameters(agent, self.configs)

        optimizer = agent.configure_optimizers(self.configs["weight_decay"], self.learning_rate, (0.9, 0.95))

        for step in tqdm(range(self.n_steps)):
            sample_kwargs = {
                'nucleus_p': self.configs['nucleus_p'],
                'use_temperature_annealing': self.configs['use_temperature_annealing'],
                'min_temp': self.configs['min_temperature'],
                'max_temp': self.configs['max_temperature'],
                'use_diversity_sampling': self.configs['use_diversity_loss']
            }
            samples, seqs, _ = sample_SMILES(agent, self.voc, n_mols=self.batch_size, **sample_kwargs)

            donor_smi = samples if self.target_type == "donor" else [self.fixed_smiles] * len(samples)
            acceptor_smi = [self.fixed_smiles] * len(samples) if self.target_type == "donor" else samples

            opv_probs = None
            if self.configs.get("use_opv_predictor", True):
                _, opv_probs = self.opv_classifier.predict(samples)

            # Get PCE back to real scale using the same de-normalization as moe2_p3/inference.py
            pce_scores = predict_pce(
                donor_smi,
                acceptor_smi,
                self.pce_model,
                self.device,
                y_mean=self.pce_mean,
                y_std=self.pce_std,
            )
            final_scores = self._calculate_reward(pce_scores, opv_probs)

            samples, pce_scores, final_scores, seqs = self._memory_update(samples, pce_scores, final_scores, seqs)

            # Calculate validity
            valid_ratio, _ = evaluate_smiles_validity(samples)
            
            # Calculate top rewards for current batch
            sorted_rewards = np.sort(final_scores)[::-1]  # Sort descending
            top1_reward = sorted_rewards[0] if len(sorted_rewards) > 0 else 0.0
            top10_reward = sorted_rewards[:10].mean() if len(sorted_rewards) >= 10 else sorted_rewards.mean()
            top100_reward = sorted_rewards[:100].mean() if len(sorted_rewards) >= 100 else sorted_rewards.mean()

            reward_tensor = to_tensor(final_scores)
            prior_lik = likelihood(prior, seqs)
            agent_lik = likelihood(agent, seqs)

            loss = torch.pow(self.sigma1 * (1 - step / self.n_steps) * reward_tensor - (prior_lik - agent_lik), 2)
            loss = loss.mean()

            if self.configs.get('use_homo_lumo_loss', False):
                preds = predict_homo_lumo(samples, self.homo_lumo_model, self.device)
                kl_div = self._kl_divergence([p[0] for p in preds], [p[1] for p in preds])
                loss += self.kl_coeff * torch.tensor(kl_div, device=self.device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self.writer:
                self.writer.add_scalar("agent/loss", loss.item(), step)
                self.writer.add_scalar("agent/avg_pce", pce_scores.mean(), step)
                self.writer.add_scalar("agent/avg_reward", final_scores.mean(), step)
                self.writer.add_scalar("agent/validity", valid_ratio, step)
                self.writer.add_scalar("agent/top1_reward", top1_reward, step)
                self.writer.add_scalar("agent/top10_reward", top10_reward, step)
                self.writer.add_scalar("agent/top100_reward", top100_reward, step)

            if self.writer and not self.memory.empty:
                self.writer.add_scalar("memory/top1_pce", self.memory["pce_scores"].iloc[0], step)
                self.writer.add_scalar("memory/avg_pce", self.memory["pce_scores"].mean(), step)
                # Log top 10 and top 100 PCE scores
                sorted_pce = self.memory["pce_scores"].sort_values(ascending=False)
                if len(sorted_pce) >= 10:
                    self.writer.add_scalar("memory/top10_pce", sorted_pce.head(10).mean(), step)
                if len(sorted_pce) >= 100:
                    self.writer.add_scalar("memory/top100_pce", sorted_pce.head(100).mean(), step)
                # Log top 1, 10, 100 rewards from memory
                sorted_rewards_mem = self.memory["scores"].sort_values(ascending=False)
                self.writer.add_scalar("memory/top1_reward", sorted_rewards_mem.iloc[0], step)
                if len(sorted_rewards_mem) >= 10:
                    self.writer.add_scalar("memory/top10_reward", sorted_rewards_mem.head(10).mean(), step)
                if len(sorted_rewards_mem) >= 100:
                    self.writer.add_scalar("memory/top100_reward", sorted_rewards_mem.head(100).mean(), step)
                if (step + 1) % 20 == 0:
                    self.writer.add_scalar("memory/diversity", int_div(list(self.memory["smiles"])), step)

            if (step + 1) % 100 == 0:
                self.memory.to_csv(os.path.join(self.configs["mem_save_path"], f"{self.configs['run_name']}_step{step+1}.csv"), index=False)
                torch.save(agent.state_dict(), os.path.join(ckpt_dir, f"step{step+1}.pt"))

        self.memory.to_csv(os.path.join(self.configs["mem_save_path"], f"{self.configs['run_name']}_final.csv"), index=False)
        if self.writer: self.writer.close()

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    trainer = CycleChemistTrainer(vars(args))
    trainer.train()
