import os, sys, time, torch, torch.nn as nn, torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from neuralop.losses import H1Loss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

def detect_model_type(model):
    model.eval()
    dummy = torch.randn(1,1,1,23,23,23, device=next(model.parameters()).device)
    out   = model(dummy)
    model.train()
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
    device, epochs, optimizer_params,
    target_face, log_dir, model_save_dir, mainloss):

    model_type = detect_model_type(model)
    run_name   = f'{model.name}_{time.strftime("%Y%m%d-%H%M%S")}'
    writer     = SummaryWriter(f'{log_dir}/{run_name}')
    model, best_loss = model.to(device), float('inf')

    if optimizer_params is None: optimizer_params = {'lr': 1e-3}
    opt  = optim.AdamW(model.parameters(), **optimizer_params)
    sch = CosineAnnealingWarmRestarts(opt, T_0=int(epochs*0.1), T_mult=9, eta_min=5e-6)

    # loss definitions
    h1fn = H1Loss(d=2, reduction='mean')
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

            mx = data.amax((-3,-2,-1), keepdim=True);  mx[mx==0]=1
            data = data / mx

            # ---------------- forward & loss -------------------------------
            if mainloss.lower() == "2head":
                out = model(data, inference=False)
                sign = model.sign_tensor
            else:
                out = model(data)

            if model_type=="FacePredictor":
                full = tgt["greens_tensor"] if "greens_tensor" in tgt else tgt["normalised_gradient"]
                full = full[:, target_face, 0]      # (B,H,W)

                if out.ndim==3: out = out.unsqueeze(1)
                if mainloss.lower() == "mse":
                    main_loss = mse(out.squeeze(1), full)
                elif mainloss.lower() == "h1":
                    main_loss = h1fn(out, full.unsqueeze(1))
                elif mainloss.lower() == "l2":
                    main_loss = l2(out, full.unsqueeze(1))
                elif mainloss.lower() == "mix1":
                    full_pdf = full.abs() + 1e-10
                    full_pdf = full_pdf / full_pdf.sum(dim=[1,2], keepdim=True)
                    pred_pdf = out.squeeze(1).abs() + 1e-10
                    loss1 = kl_div(pred_pdf.log(), full_pdf)
                    # loss2 = mse(out.squeeze(1), full)
                    sign_target = (full >= 0).float()  # shape: (B, h, w)
                    loss2 = torch.nn.functional.binary_cross_entropy_with_logits(out.squeeze(1), sign_target, reduction='mean')
                    main_loss = loss1 + loss2
                elif mainloss.lower() == "mix2":
                    full_pdf = full.abs() + 1e-10
                    full_pdf = full_pdf / full_pdf.sum(dim=[1,2], keepdim=True)
                    pred_pdf = out.squeeze(1).abs() + 1e-10
                    loss1 = kl_div(pred_pdf.log(), full_pdf)
                    # loss2 = mse(out.squeeze(1), full)
                    sign_target = (full >= 0).float()  # shape: (B, h, w)
                    loss2 = torch.nn.functional.binary_cross_entropy_with_logits(23*23*out.squeeze(1), sign_target, reduction='mean')
                    main_loss = loss1 + loss2
                elif mainloss.lower() == "2head":
                    full_pdf = full.abs() + 1e-10
                    full_pdf = full_pdf / full_pdf.sum(dim=[1,2], keepdim=True)
                    loss1 = kl_div(out.squeeze(1).log(), full_pdf)
                    # loss2 = mse(out.squeeze(1), full)
                    sign_target = (full >= 0).float()  # shape: (B, h, w)
                    loss2 = torch.nn.functional.binary_cross_entropy_with_logits(sign, sign_target, reduction='mean')
                    main_loss = loss1 + loss2
                else:
                    full_pdf = full.abs() + 1e-10
                    full_pdf = full_pdf / full_pdf.sum(dim=[1,2], keepdim=True)
                    main_loss = kl_div(out.squeeze(1).log(), full_pdf)

            elif model_type=="FaceSelector3D":
                faces = tgt["face_distribution"]
                main_loss = kl_div(out[:, :6].log(), faces)

            else:   # FaceSelectorWeight3D
                faces = tgt["face_grad_weights"]                # (B,7)
                kl_div_loss = kl_div(out[:, :6].log(), faces[:,:6])
                mse_loss = mse(out[:,6], faces[:,6])
                kl_weight = 1 # float(mainloss)
                main_loss = kl_weight * kl_div_loss + mse_loss


            main_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            writer.add_scalar('train/learning_rate', opt.param_groups[0]['lr'], global_step)
            sch.step(ep + n_batches/len(train_loader))

            ep_loss += main_loss.item()
            writer.add_scalar('train/batch_loss', main_loss.item(), global_step)
            global_step += 1; n_batches += 1

        ep_loss /= n_batches
        writer.add_scalar('train/epoch_loss', ep_loss, ep)
        print(f"    ⮕  train_loss={ep_loss:.6f}")

        # ---------------- validation / test -------------------------------
        val_loss = test_model_improved(
            model, val_loader, device, target_face,
            model_type, writer, ep, mainloss=mainloss)
        print(f"    ⮕  test_loss={val_loss:.6f}")
        sys.stdout.flush()

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

def output_vs_truth(out, full, writer, metric_dict):
    import matplotlib.pyplot as plt
    
    # Take first sample from batch for visualization
    sample_out = out[0].squeeze().cpu().numpy()
    sample_full = full[0].cpu().numpy()
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Find common scale for both plots
    vmin = min(sample_out.min(), sample_full.min())
    vmax = max(sample_out.max(), sample_full.max())
    
    # Plot prediction
    im1 = ax1.imshow(sample_out, cmap='viridis', vmin=vmin, vmax=vmax)
    ax1.set_title(f'Prediction (Epoch {metric_dict['epoch']})')
    plt.colorbar(im1, ax=ax1)
    
    # Plot ground truth
    im2 = ax2.imshow(sample_full, cmap='viridis', vmin=vmin, vmax=vmax)
    ax2.set_title('Ground Truth')
    plt.colorbar(im2, ax=ax2)
    
    # Plot difference
    diff = sample_out - sample_full
    im3 = ax3.imshow(diff, cmap='RdBu_r')
    ax3.set_title('Difference')
    plt.colorbar(im3, ax=ax3)

    # Add metrics as text annotations
    ax1.text(0.02, 0.98, f'MSE: {metric_dict["mse_loss"]:.4e}', 
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.text(0.02, 0.98, f'H1: {metric_dict["h1_loss"]:.4e}', 
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax3.text(0.02, 0.98, f'L2: {metric_dict["l2_loss"]:.4e}\nKL: {metric_dict["kl_div_loss"]:.4e}', 
             transform=ax3.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if writer:
        writer.add_figure('validation/prediction_vs_truth', fig, metric_dict['epoch'])
    
    plt.close()

def output_vs_truth_selector(out, full, writer, metric_dict, n=6):
    import matplotlib.pyplot as plt
    # out[:, :6] = torch.exp(out[:, :6])
    
    # Take first n samples from batch for visualization
    sample_out = out[:n].cpu().numpy()  # shape: (n, 6)
    sample_full = full[:n].cpu().numpy()  # shape: (n, 6)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(min(n, len(axes))):
        ax = axes[i]
        
        # Create x-axis for bar plots
        x = range(out.size(1))
        
        # Plot comparison for sample i
        width = 0.35
        x_shifted = [j - width/2 for j in x]
        x_shifted2 = [j + width/2 for j in x]
        ax.bar(x_shifted, sample_out[i], width, label='Prediction', alpha=0.7)
        ax.bar(x_shifted2, sample_full[i], width, label='Ground Truth', alpha=0.7, color='orange')
        ax.set_title(f'Sample {i+1} (Epoch {metric_dict["epoch"]})')
        ax.set_xlabel('Face Index')
        ax.set_ylabel('Probability')
        ax.set_xticks(x)
        ax.legend()
        
        # Add all metrics from metric_dict as text annotation
        metrics_text = '\n'.join([f'{k}: {v:.4e}' if isinstance(v, float) else f'{k}: {v}' 
                     for k, v in metric_dict.items() if k != 'epoch'])
        ax.text(0.02, 0.98, metrics_text, 
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide unused subplots
    for i in range(n, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if writer:
        writer.add_figure('validation/selector_distribution', fig, metric_dict['epoch'])
    
    plt.close()

# ─────────────────────────────── test fn (minor tweak) ─────────────────────────
def test_model_improved(model, loader, device, target_face,
                        model_type, writer=None, epoch=None, mainloss="kl_div"):
    model.eval()
    

    # loss definitions
    h1fn = H1Loss(d=2, reduction='mean')
    mse = nn.MSELoss()
    kl_div = nn.KLDivLoss(reduction='batchmean')
    l2 = compute_relative_l2_error
    mare = lambda x, y: torch.mean(torch.abs(x - y) / (torch.abs(y) + 1e-10))    
    tot_main_loss = 0; main_loss = 0 
    tot_h1_loss = 0; h1_loss = 0
    tot_mse_loss = 0; mse_loss = 0
    tot_kl_div_loss = 0; kl_div_loss = 0
    tot_l2_loss = 0; l2_loss = 0
    tot_mare_loss = 0; mare_loss = 0
    tot_bce_loss = 0; bce_loss = 0
    tot_scaled_bce_loss = 0; scaled_bce_loss = 0
    tot_signhead_loss = 0; signhead_loss = 0
    n = 0
    finish_plot = False

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
                
                full_pdf = full.abs() + 1e-10
                full_pdf = full_pdf / full_pdf.sum(dim=[1,2], keepdim=True)
                pred_pdf = out.squeeze(1).abs() + 1e-10

                kl_div_loss = kl_div(pred_pdf.log(), full_pdf)
                sign_target = (full >= 0).float()  # shape: (B, h, w)
                bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(out.squeeze(1), sign_target, reduction='mean')
                scaled_bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(23*23*out.squeeze(1), sign_target, reduction='mean')
                if epoch is not None and epoch % 10 == 0 and not finish_plot:
                    metric_dict = {
                        'mse_loss': mse_loss.item(),
                        'h1_loss': h1_loss.item(),
                        'l2_loss': l2_loss.mean().item(),
                        'kl_div_loss': kl_div_loss.item(),
                        'epoch': epoch,
                    }
                    output_vs_truth(out, full, writer, metric_dict)
                    finish_plot = True

                if mainloss.lower() == "mse":
                    main_loss = mse_loss
                elif mainloss.lower() == "h1":
                    main_loss = h1_loss
                elif mainloss.lower() == "l2":
                    main_loss = l2_loss.mean()
                elif mainloss.lower() == "mix1":
                    loss1 = kl_div_loss
                    # loss2 = mse_loss
                    loss2 = bce_loss
                    main_loss = loss1 + loss2
                elif mainloss.lower() == "mix2":
                    loss1 = kl_div_loss
                    # loss2 = mse_loss
                    loss2 = scaled_bce_loss
                    main_loss = loss1 + loss2
                elif mainloss.lower() == "2head":
                    loss1 = kl_div_loss
                    signhead_loss = torch.nn.functional.binary_cross_entropy_with_logits(model.sign_tensor, sign_target, reduction='mean')
                    main_loss = loss1 + signhead_loss
                else:
                    main_loss = kl_div_loss

            elif model_type=="FaceSelector3D":
                faces = tgt["face_distribution"]
                kl_div_loss = kl_div(out[:, :6].log(), faces)
                main_loss = kl_div_loss
                if epoch is not None and epoch % 10 == 0 and not finish_plot:
                    metric_dict = {
                        'kl_div_loss': kl_div_loss.item(),
                        'epoch': epoch,
                    }
                    output_vs_truth_selector(out, faces, writer, metric_dict)
                    finish_plot = True

            else:   # FaceSelectorWeight3D
                faces = tgt["face_grad_weights"]                # (B,7)
                kl_div_loss = kl_div(out[:, :6].log(), faces[:,:6])
                mse_loss = mse(out[:,6], faces[:,6])
                mare_loss = mare(out[:,6], faces[:,6])
                kl_weight = 1 # float(mainloss)
                main_loss = kl_weight * kl_div_loss + mse_loss
                if epoch is not None and epoch % 10 == 0 and not finish_plot:
                    metric_dict = {
                        'kl_div_loss': kl_div_loss.item(),
                        'mse_loss': mse_loss.item(),
                        'mare_loss': mare_loss.item(),
                        'main_loss': main_loss.item(),
                        'epoch': epoch,
                    }
                    output_vs_truth_selector(out, faces, writer, metric_dict)
                    finish_plot = True
               
            tot_main_loss += main_loss.item(); n += 1
            tot_h1_loss += h1_loss.item() if model_type=="FacePredictor" else 0
            tot_mse_loss += mse_loss.item() if model_type!="FaceSelector3D" else 0
            tot_kl_div_loss += kl_div_loss.item()
            tot_l2_loss += l2_loss.mean().item() if model_type=="FacePredictor" else 0
            tot_mare_loss += mare_loss.item() if model_type=="FaceSelectorWeight3D" else 0
            tot_bce_loss += bce_loss.item() if model_type=="FacePredictor" else 0
            tot_scaled_bce_loss += scaled_bce_loss.item() if model_type=="FacePredictor" else 0
            tot_signhead_loss += signhead_loss.item() if model_type=="FacePredictor" and mainloss.lower() == "2head" else 0

    if writer and epoch is not None:
        writer.add_scalar('test/epoch_loss', tot_main_loss/n, epoch)
        if model_type == "FacePredictor":
            writer.add_scalar('test/h1_loss', tot_h1_loss/n, epoch)
            writer.add_scalar('test/mse_loss', tot_mse_loss/n, epoch)
            writer.add_scalar('test/l2_loss', tot_l2_loss/n, epoch)
            writer.add_scalar('test/kl_div_loss', tot_kl_div_loss/n, epoch)
            writer.add_scalar('test/bce_loss', tot_bce_loss/n, epoch)
            writer.add_scalar('test/scaled_bce_loss', tot_scaled_bce_loss/n, epoch)
            writer.add_scalar('test/signhead_loss', tot_signhead_loss/n, epoch)
        else:
            writer.add_scalar('test/kl_div_loss', tot_kl_div_loss/n, epoch)
            if model_type == "FaceSelectorWeight3D":
                writer.add_scalar('test/mare_loss', tot_mare_loss/n, epoch)
                writer.add_scalar('test/mse_loss', tot_mse_loss/n, epoch)
    return tot_main_loss/n
