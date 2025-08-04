# train.py  •  2025-07-05
#
# Only the parts interacting with targets changed — all logic,
# metrics, H1-loss handling, etc. remain intact.

import os, time, torch, torch.nn as nn, torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from neuralop.losses import H1Loss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from architecture import *

# ───────────────────────── model-shape detector (unchanged) ─────────────────────
def detect_model_type(model):
    dummy = torch.randn(1,1,1,23,23,23, device=next(model.parameters()).device)
    out   = model(dummy)
    if out.ndim==3 and out.shape[1]==23:              return "FacePredictor"
    if out.ndim==4 and out.shape[1]==1:               return "FacePredictor"
    if out.ndim==2 and out.shape[1]==6:               return "FaceSelector3D"
    if out.ndim==2 and out.shape[1]==7:               return "FaceSelectorWeight3D"
    raise ValueError(f"Unknown output shape {out.shape}")

def compute_relative_l2_error(prediction, target):
    """Compute relative L2 error."""
    diff = prediction - target
    l2_error = torch.sqrt(torch.sum(diff**2, dim=(-2, -1)))
    l2_target = torch.sqrt(torch.sum(target**2, dim=(-2, -1)))
    relative_error = l2_error / (l2_target + 1e-8)
    return relative_error

# ─────────────────────────────── training fn ───────────────────────────────────
def train_model_improved(
    model, train_loader, val_loader,
    device='cuda', epochs=50, optimizer_params=None,
    target_face=None, log_dir='../runs', model_save_dir='./trained_models'):

    model_type = detect_model_type(model)
    run_name   = f'{model.name}_{time.strftime("%Y%m%d-%H%M%S")}'
    writer     = SummaryWriter(f'{log_dir}/{run_name}')
    model, best_loss = model.to(device), float('inf')

    if optimizer_params is None: optimizer_params = {'lr': 1e-3}
    opt  = optim.Adam(model.parameters(), **optimizer_params)
    sch = CosineAnnealingWarmRestarts(opt, T_0=epochs//2, T_mult=2, eta_min=5e-6)

    # loss definitions
    h1fn = H1Loss(d=2)
    mse = nn.MSELoss()
    kl_div = nn.KLDivLoss(reduction='batchmean')
    l2 = compute_relative_l2_error
    mare = lambda x, y: torch.mean(torch.abs(x - y) / (torch.abs(y) + 1e-8))    
    ep_loss = 0; main_loss = 0; h1_loss = 0; mse_loss = 0; kl_div_loss = 0

    print(f"▶️  {model.name}: type={model_type}, epochs={epochs}")
    global_step = 0
    for ep in range(epochs):
        model.train(); ep_loss = 0; n_batches = 0
        print(f"  Epoch {ep+1}/{epochs}")

        for data, tgt in train_loader:                         # <<< tgt is a dict
            data = data.to(device)
            tgt  = {k:v.to(device) for k,v in tgt.items()}
            opt.zero_grad()

            # optional normalisation (unchanged)
            mx = data.amax((-3,-2,-1), keepdim=True);  mx[mx==0]=1
            data = data / mx

            # ---------------- forward & loss -------------------------------
            out = model(data)

            if model_type=="FacePredictor":
                full = tgt["greens_tensor"] if "greens_tensor" in tgt else tgt["normalised_gradient"]
                full = full[:, target_face, 0]      # (B,H,W)

                if out.ndim==3: out = out.unsqueeze(1)

                mse_loss = mse(out.squeeze(1), full)
                h1_loss = h1fn(out, full.unsqueeze(1))
                main_loss = mse_loss + h1_loss * 0.1

            elif model_type=="FaceSelector3D":
                faces = tgt["face_distribution"]
                main_loss = kl_div(out[:, :6], faces)

            else:   # FaceSelectorWeight3D
                faces = tgt["face_grad_weights"]                # (B,7)
                kl_div_loss = kl_div(out[:, :6], faces[:,:6])
                mse_loss = mse(out[:,6], faces[:,6])
                main_loss = kl_div_loss + mse_loss

            main_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            sch.step()

            ep_loss += main_loss.item()
            writer.add_scalar('train/batch_loss', main_loss.item(), global_step)
            global_step += 1; n_batches += 1

        ep_loss /= n_batches
        writer.add_scalar('train/epoch_loss', ep_loss, ep)
        print(f"    ⮕  train_loss={ep_loss:.6f}")

        # ---------------- validation / test -------------------------------
        val_loss = test_model_improved(
            model, val_loader, device, target_face,
            model_type, writer, ep)
        print(f"    ⮕  test_loss={val_loss:.6f}")

        # best-model tracking
        if val_loss < best_loss:
            best_loss = val_loss
            os.makedirs(model_save_dir, exist_ok=True)
            pt = os.path.join(model_save_dir, f"{model.name}_best.pt")
            torch.save(model.state_dict(), pt)

            jit = os.path.join(model_save_dir, f"{model.name}_best.jit")
            torch.jit.save(torch.jit.script(model), jit)
            print(f"      💾 saved new best → {jit}")

    writer.close()
    return model, best_loss

# ─────────────────────────────── test fn (minor tweak) ─────────────────────────
def test_model_improved(model, loader, device, target_face,
                        model_type, writer=None, epoch=None):
    model.eval()
    

    # loss definitions
    h1fn = H1Loss(d=2)
    mse = nn.MSELoss()
    kl_div = nn.KLDivLoss(reduction='batchmean')
    l2 = compute_relative_l2_error
    mare = lambda x, y: torch.mean(torch.abs(x - y) / (torch.abs(y) + 1e-8))    
    tot_main_loss = 0; main_loss = 0 
    tot_h1_loss = 0; h1_loss = 0
    tot_mse_loss = 0; mse_loss = 0
    tot_kl_div_loss = 0; kl_div_loss = 0
    tot_l2_loss = 0; l2_loss = 0
    tot_mare_loss = 0; mare_loss = 0
    n = 0

    with torch.no_grad():
        for data, tgt in loader:
            data = data.to(device)
            tgt  = {k:v.to(device) for k,v in tgt.items()}
            mx = data.amax((-3,-2,-1), keepdim=True); mx[mx==0]=1
            data = data / mx
            out  = model(data)

            if model_type=="FacePredictor":
                full = tgt["greens_tensor"] if "greens_tensor" in tgt else tgt["normalised_gradient"]
                full = full[:, target_face, 0]      # (B,H,W)

                if out.ndim==3: out = out.unsqueeze(1)

                mse_loss = mse(out.squeeze(1), full)
                h1_loss = h1fn(out, full.unsqueeze(1))
                l2_loss = l2(out, full.unsqueeze(1))
                main_loss = mse_loss + h1_loss * 0.1

            elif model_type=="FaceSelector3D":
                faces = tgt["face_distribution"]
                kl_div_loss = kl_div(out[:, :6], faces)
                main_loss = kl_div_loss

            else:   # FaceSelectorWeight3D
                faces = tgt["face_grad_weights"]                # (B,7)
                kl_div_loss = kl_div(out[:, :6], faces[:,:6])
                mse_loss = mse(out[:,6], faces[:,6])
                mare_loss = mare(out[:,6], faces[:,6])
                main_loss = kl_div_loss + mse_loss
               
            tot_main_loss += main_loss.item(); n += 1
            tot_h1_loss += h1_loss.item() if model_type=="FacePredictor" else 0
            tot_mse_loss += mse_loss.item() if model_type!="FaceSelector3D" else 0
            tot_kl_div_loss += kl_div_loss.item() if model_type!="FacePredictor" else 0
            tot_l2_loss += l2_loss.mean().item() if model_type=="FacePredictor" else 0
            tot_mare_loss += mare_loss.item() if model_type=="FaceSelectorWeight3D" else 0

    if writer and epoch is not None:
        writer.add_scalar('test/epoch_loss', tot_main_loss/n, epoch)
        if model_type == "FacePredictor":
            writer.add_scalar('test/h1_loss', tot_h1_loss/n, epoch)
            writer.add_scalar('test/mse_loss', tot_mse_loss/n, epoch)
            writer.add_scalar('test/l2_loss', tot_l2_loss/n, epoch)
        else:
            writer.add_scalar('test/kl_div_loss', tot_kl_div_loss/n, epoch)
            if model_type == "FaceSelectorWeight3D":
                writer.add_scalar('test/mare_loss', tot_mare_loss/n, epoch)
                writer.add_scalar('test/mse_loss', tot_mse_loss/n, epoch)
    return tot_main_loss/n
