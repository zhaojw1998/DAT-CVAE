import torch
from torch import nn
import os
import sys
sys.path.append('./harmonization_model_ver8')
import json
import numpy as np
import utils
from torch import optim
import torch.nn.functional as F
from utils import MinExponentialLR, loss_function_vae, loss_function_discr, scheduled_sampling, AverageMeter
from model import VAE
from torch.utils.tensorboard import SummaryWriter
from nottingham_dataset import Nottingham
import time
from torch.utils.data import DataLoader
import platform


DEBUG_MODE = (platform.system() == 'Windows')
###############################################################################
# Load config
###############################################################################
config_fn = './harmonization_model_ver8/model_config.json'
train_hyperparams = utils.load_params_dict('train_hyperparams', config_fn)
model_params = utils.load_params_dict('model_params', config_fn)
data_repr_params = utils.load_params_dict('data_repr', config_fn)
project_params = utils.load_params_dict('project', config_fn)
dataset_paths = utils.load_params_dict('dataset_paths', config_fn)

BATCH_SIZE = train_hyperparams['batch_size']
LEARNING_RATE = train_hyperparams['learning_rate']
DECAY = train_hyperparams['decay']
PARALLEL = train_hyperparams['parallel']
if sys.platform in ['win32', 'darwin']:
    PARALLEL = False
N_EPOCH = train_hyperparams['n_epoch']
CLIP = train_hyperparams['clip']
UP_AUG = train_hyperparams['up_aug']
DOWN_AUG = train_hyperparams['down_aug']
INIT_WEIGHT = train_hyperparams['init_weight']
WEIGHTS = tuple(train_hyperparams['weights'])
TFR1 = tuple(train_hyperparams['teacher_forcing1'])
TFR2 = tuple(train_hyperparams['teacher_forcing2'])

###############################################################################
# Initialize project
###############################################################################
PROJECT_NAME = project_params['project_name']

if DEBUG_MODE:
    save_path = project_params['laptop_save_root']
    RUN_TIME = time.strftime("%Y%m%d-%H-%M-%S", time.localtime())
    project_path = os.path.join(save_path, 'debug_' + RUN_TIME)
    dataset_path = dataset_paths['laptop_data_path']
else:
    save_path = project_params['hpc_save_root']
    RUN_TIME = time.strftime("%Y%m%d-%H-%M-%S", time.localtime())
    project_path = os.path.join(save_path, RUN_TIME + '_' + PROJECT_NAME)
    dataset_path = dataset_paths['hpc_data_path']
    
MODEL_PATH = os.path.join(project_path, project_params['model_path']) 
LOG_PATH = os.path.join(project_path, project_params['log_path'])

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)


decay_writer = SummaryWriter(os.path.join(LOG_PATH, 'decay'))
batch_loss_writer = SummaryWriter(os.path.join(LOG_PATH, 'batch_loss'))
epoch_loss_writer = SummaryWriter(os.path.join(LOG_PATH, 'epoch_loss'))

print('Project initialized.', flush=True)

###############################################################################
# load data
###############################################################################
dataset = np.load(dataset_path, allow_pickle=True).T
np.random.seed(0)
np.random.shuffle(dataset)
#anchor = int(dataset.shape[0] * 0.95)
if DEBUG_MODE:
    anchor = int(dataset.shape[0] * 0.002)
    train_data = dataset[:anchor, :]
    anchor = int(dataset.shape[0] * 0.998)
    val_data = dataset[anchor:, :]
    BATCH_SIZE = 32
else:
    anchor = int(dataset.shape[0] * 0.95)
    train_data = dataset[:anchor, :]
    val_data = dataset[anchor:, :]
train_data = train_data.T
val_data = val_data.T
train_dataset = Nottingham(train_data, 128, 32, 'pr', mask_ratio=1., shift_low=-5, shift_high=6)
val_dataset = Nottingham(val_data, 128, 32, 'pr', mask_ratio=1., shift_low=-5, shift_high=6)


val_loader = DataLoader(val_dataset, BATCH_SIZE*4, False)

print('Batch Size:', BATCH_SIZE, 'Train Batches:', len(train_dataset) // BATCH_SIZE, 'Val Batches:', len(val_dataset) // BATCH_SIZE, flush=True)
print('Dataset loaded!', flush=True)

###############################################################################
# model parameter
###############################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VAE(max_simu_note=data_repr_params['max_simu_note'],
            max_pitch=data_repr_params['max_pitch'],
            min_pitch=data_repr_params['min_pitch'],
            pitch_sos=data_repr_params['pitch_sos'],
            pitch_eos=data_repr_params['pitch_eos'],
            pitch_pad=data_repr_params['pitch_pad'],
            pitch_hold=data_repr_params['pitch_hold'],
            pitch_rest=data_repr_params['pitch_rest'],
            pitch_mask=data_repr_params['pitch_mask'],
            num_step=data_repr_params['num_time_step'],

            note_emb_size=model_params['note_emb_size'],
            enc_notes_hid_size=model_params['enc_notes_hid_size'],
            enc_time_hid_size=model_params['enc_time_hid_size'],
            z_size=model_params['z_size'],
            dec_emb_hid_size=model_params['dec_emb_hid_size'],
            dec_time_hid_size=model_params['dec_time_hid_size'],
            dec_notes_hid_size=model_params['dec_notes_hid_size'],
            discr_nhead = model_params["discr_nhead"],
            discr_hid_size = model_params["discr_hid_size"], 
            discr_dropout = model_params["discr_dropout"], 
            discr_nlayer = model_params["discr_nlayer"],

            device=device
            )

if INIT_WEIGHT:
    model.apply(utils.init_weights)
    print('The parameters in the model are initialized!')
print(f'The model has {utils.count_parameters(model):,} trainable parameters')

if PARALLEL:
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model = model.to(device)
    model = model.module
else:
    model = model.to(device)
print('Model loaded!')

###############################################################################
# Optimizer and Criterion
###############################################################################
optimizer_enc = optim.Adam(model.encoder_params(), lr=LEARNING_RATE)
optimizer_dec = optim.Adam(model.decoder_params(), lr=LEARNING_RATE)
optimizer_discr = optim.Adam(model.discr_params(), lr=LEARNING_RATE)

pitch_loss_func = nn.CrossEntropyLoss(ignore_index=model.pitch_pad)
mask_loss_func = nn.BCEWithLogitsLoss()
normal = utils.standard_normal(model.z_size)
if DECAY:
    scheduler_enc = MinExponentialLR(optimizer_enc, gamma=0.99999, minimum=1e-5)
    scheduler_dec = MinExponentialLR(optimizer_dec, gamma=0.99995, minimum=1e-5)
    scheduler_discr = MinExponentialLR(optimizer_discr, gamma=0.9999, minimum=1e-6)

mse = torch.nn.MSELoss(reduction ='mean')

if DEBUG_MODE:
    N = 100
else:
    N = 100


vae_pitch_loss_meter = AverageMeter()
vae_mask_loss = AverageMeter()
vae_kl_loss = AverageMeter()

discr_pitch_loss = AverageMeter()
discr_mask_discr_loss = AverageMeter()
discr_mask_enc_loss = AverageMeter()
discr_kl_loss = AverageMeter()

###############################################################################
# Main
###############################################################################
def train(model, train_loader,
          pitch_criterion, mask_criterion, normal, weights,
          decay, clip, epoch):
    #epoch_loss = 0.
    epoch_vae_pitch_loss = 0.
    epoch_vae_mask_loss = 0.
    epoch_discr_pitch_loss = 0.
    epoch_discr_mask_loss_1 = 0.
    epoch_discr_mask_loss_2 = 0.
    epoch_kl_loss = 0.

    train_vae = 0.
    train_discr_upd_discr = 0.
    train_discr_upd_enc = 0.
    
    
    
    num_batch = len(train_loader)

    for i, (chord, melody, masked_melody) in enumerate(train_loader):
        if torch.cuda.is_available():
            chord = chord.long().cuda()
            melody = melody.float().cuda()
            masked_melody = masked_melody.float().cuda()

        optimizer_enc.zero_grad()
        optimizer_dec.zero_grad()
        optimizer_discr.zero_grad()

        if i % 20 < 10:
            tfr1 = scheduled_sampling(((epoch + i / num_batch) / N_EPOCH),
                                    TFR1[0], TFR1[1])
            tfr2 = scheduled_sampling(((epoch + i / num_batch) / N_EPOCH),
                                    TFR2[0], TFR2[1])
        
        recon_pitch, dist, mask_prediction, masked_idx = model(chord, melody, masked_melody, False, True, tfr1, tfr2)

        recon_pitch = recon_pitch.view(-1, recon_pitch.size(-1))
        mask_gt = melody[masked_idx[0], masked_idx[1], :12]

        #VAE、Discr交替训练100次，Discr训练时交替update discr和enc
        if i % 20 < 10:
            #train and update vae
            loss, pitch_loss, kl_loss = \
                loss_function_vae(recon_pitch,
                                chord[:, :, 1:, 0].contiguous().view(-1),
                                dist, 
                                pitch_criterion, 
                                normal,
                                weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer_enc.step()
            optimizer_dec.step()
            if decay:
                scheduler_enc.step()
                scheduler_dec.step()

            with torch.no_grad():
                loss, mask_loss, kl_loss = \
                loss_function_discr(mask_prediction, 
                                mask_gt, 
                                dist, 
                                mask_criterion, 
                                normal, 
                                weights)
            
            epoch_vae_pitch_loss += pitch_loss.item()
            epoch_vae_mask_loss += mask_loss.item()
            epoch_kl_loss += kl_loss.item()

            vae_pitch_loss_meter.update(pitch_loss.item())
            vae_mask_loss.update(mask_loss.item())
            vae_kl_loss.update(kl_loss.item())

            batch_loss_writer.add_scalar('train_vae/pitch_loss', vae_pitch_loss_meter.avg,
                               epoch * num_batch + i)
            batch_loss_writer.add_scalar('train_vae/mask_loss', vae_mask_loss.avg,
                               epoch * num_batch + i)
            batch_loss_writer.add_scalar('train_vae/kl_loss', vae_kl_loss.avg,
                               epoch * num_batch + i)
            if DEBUG_MODE:
                print('------------train vae------------', flush=True)
                print('Epoch: [{0}][{1}/{2}]'.format(epoch+1, i, num_batch), flush=True)
                print('ploss: {ploss:.5f}, mloss: {mloss:.5f}, klloss: {klloss:.5f}'.format(ploss=pitch_loss.item(), mloss=mask_loss.item(), klloss=kl_loss.item()), flush=True)
            train_vae += 1

            
        
        elif i % 20 < 15:
            #train discr and update discr
            loss, mask_loss, kl_loss = \
                loss_function_discr(mask_prediction, 
                                mask_gt, 
                                dist, 
                                mask_criterion, 
                                normal, 
                                weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            #optimizer_enc.step()
            optimizer_discr.step()
            if decay:
                #scheduler_enc.step()
                scheduler_discr.step()

            with torch.no_grad():
                loss, pitch_loss, kl_loss = \
                loss_function_vae(recon_pitch,
                            chord[:, :, 1:, 0].contiguous().view(-1),
                            dist, pitch_criterion, normal,
                            weights)
            
            epoch_discr_pitch_loss += pitch_loss.item()
            epoch_discr_mask_loss_1 += mask_loss.item()
            epoch_kl_loss += kl_loss.item()

            discr_pitch_loss.update(pitch_loss.item())
            discr_mask_discr_loss.update(mask_loss.item())
            discr_kl_loss.update(kl_loss.item())

            batch_loss_writer.add_scalar('train_discr/pitch_loss', discr_pitch_loss.avg,
                               epoch * num_batch + i)
            batch_loss_writer.add_scalar('train_discr/mask_loss_upd_discr', discr_mask_discr_loss.avg,
                               epoch * num_batch + i)
            batch_loss_writer.add_scalar('train_discr/kl_loss', discr_kl_loss.avg,
                               epoch * num_batch + i)
            if DEBUG_MODE:
                print('----train discr, update discr----', flush=True)
                print('Epoch: [{0}][{1}/{2}]'.format(epoch+1, i, num_batch), flush=True)
                print('ploss: {ploss:.5f}, mloss: {mloss:.5f}, klloss: {klloss:.5f}'.format(ploss=pitch_loss.item(), mloss=mask_loss.item(), klloss=kl_loss.item()), flush=True)
            train_discr_upd_discr += 1

        else:
            #train discr and update encoder
            loss, mask_loss, kl_loss = \
                loss_function_discr(mask_prediction, 
                                1 - mask_gt, 
                                dist, 
                                mask_criterion, 
                                normal, 
                                weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer_enc.step()
            #optimizer_discr.step()
            if decay:
                scheduler_enc.step()
                #scheduler_discr.step()

            with torch.no_grad():
                loss, pitch_loss, kl_loss = \
                loss_function_vae(recon_pitch,
                            chord[:, :, 1:, 0].contiguous().view(-1),
                            dist, pitch_criterion, normal,
                            weights)
            
            epoch_discr_pitch_loss += pitch_loss.item()
            epoch_discr_mask_loss_2 += mask_loss.item()
            epoch_kl_loss += kl_loss.item()

            discr_pitch_loss.update(pitch_loss.item())
            discr_mask_enc_loss.update(mask_loss.item())
            discr_kl_loss.update(kl_loss.item())

            batch_loss_writer.add_scalar('train_discr/pitch_loss', discr_pitch_loss.avg,
                               epoch * num_batch + i)
            batch_loss_writer.add_scalar('train_discr/mask_loss_upd_enc', discr_mask_enc_loss.avg,
                               epoch * num_batch + i)
            batch_loss_writer.add_scalar('train_discr/kl_loss', discr_kl_loss.avg,
                               epoch * num_batch + i)
            if DEBUG_MODE:
                print('---train discr, update encoder---', flush=True)
                print('Epoch: [{0}][{1}/{2}]'.format(epoch+1, i, num_batch), flush=True)
                print('ploss: {ploss:.5f}, mloss: {mloss:.5f}, klloss: {klloss:.5f}'.format(ploss=pitch_loss.item(), mloss=mask_loss.item(), klloss=kl_loss.item()), flush=True)
            train_discr_upd_enc += 1

        decay_writer.add_scalar('decay_record/TFR1', tfr1, epoch * num_batch + i)
        decay_writer.add_scalar('decay_record/TFR2', tfr2, epoch * num_batch + i)
        decay_writer.add_scalar('decay_record rate/encoder_lr', optimizer_enc.param_groups[0]['lr'], epoch * num_batch + i)
        decay_writer.add_scalar('decay_record rate/decoder_lr', optimizer_dec.param_groups[0]['lr'], epoch * num_batch + i)
        decay_writer.add_scalar('decay_record rate/discr_lr', optimizer_discr.param_groups[0]['lr'], epoch * num_batch + i)

    return (epoch_vae_pitch_loss / max(train_vae, 1), \
            epoch_vae_mask_loss / max(train_vae, 1), \
            epoch_discr_pitch_loss / max(train_discr_upd_discr + train_discr_upd_enc, 1), \
            epoch_discr_mask_loss_1 / max(train_discr_upd_discr, 1), \
            epoch_discr_mask_loss_2 / max(train_discr_upd_enc, 1), \
            epoch_kl_loss / num_batch)



def evaluate(model, val_loader, pitch_criterion, mask_criterion, normal, weights, epoch):
    epoch_pitch_loss = 0.
    epoch_mask_loss = 0.
    epoch_kl_loss = 0.
    num_batch = len(val_loader)
    with torch.no_grad():
        for i, (chord, melody, masked_melody) in enumerate(val_loader):
            if torch.cuda.is_available():
                chord = chord.long().cuda()
                melody = melody.float().cuda()
                masked_melody = masked_melody.float().cuda()
            
            tfr1 = scheduled_sampling(((epoch + i / num_batch) / N_EPOCH),
                                      TFR1[0], TFR1[1])
            tfr2 = scheduled_sampling(((epoch + i / num_batch) / N_EPOCH),
                                      TFR2[0], TFR2[1])

            recon_pitch, dist, mask_prediction, masked_idx = model(chord, melody, masked_melody, False, True, tfr1, tfr2)

            recon_pitch = recon_pitch.view(-1, recon_pitch.size(-1))
            mask_gt = melody[masked_idx[0], masked_idx[1], :12]

            loss, pitch_loss, kl_loss = \
                loss_function_vae(recon_pitch,
                              chord[:, :, 1:, 0].contiguous().view(-1),
                              dist, 
                              pitch_criterion, 
                              normal,
                              weights)
            
            loss, mask_loss, kl_loss = \
                loss_function_discr(mask_prediction, 
                                mask_gt, 
                                dist, 
                                mask_criterion, 
                                normal, 
                                weights)

            epoch_pitch_loss += pitch_loss
            epoch_mask_loss += mask_loss
            epoch_kl_loss += kl_loss
            if DEBUG_MODE:
                print('-----------evaluation------------', flush=True)
                print('Epoch: [{0}][{1}/{2}]'.format(epoch+1, i, num_batch), flush=True)
                print('ploss: {ploss:.5f}, mloss: {mloss:.5f}, klloss: {klloss:.5f}'.format(ploss=pitch_loss.item(), mloss=mask_loss.item(), klloss=kl_loss.item()), flush=True)

    return (epoch_pitch_loss / num_batch, \
            epoch_mask_loss / num_batch, \
            epoch_kl_loss / num_batch)
            


for epoch in range(N_EPOCH):
    train_loader = DataLoader(train_dataset, BATCH_SIZE, True)
    print(f'Start Epoch: {epoch + 1:02}', flush=True)

    start_time = time.time()
    model.train()
    vae_ploss, vae_mloss, discr_ploss, discr_mloss_1, discr_mloss_2, kl_loss = \
        train(model, train_loader, pitch_loss_func, mask_loss_func, normal,
              WEIGHTS, DECAY, CLIP, epoch)

    model.eval()
    eval_ploss, eval_mloss, eval_kl_loss = \
        evaluate(model, val_loader,
                 pitch_loss_func, mask_loss_func, normal,
                 WEIGHTS, epoch)
    end_time = time.time()

    epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)

    #torch.save(model.state_dict(), os.path.join(MODEL_PATH,
    #                                            'pgrid-epoch-model.pt'))
    torch.save(model.state_dict(),
                os.path.join(MODEL_PATH, 'ad-ptvae_param_'+str(epoch).zfill(3)+'.pt'))
    print('Model Saved!', flush=True)

    
    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s',
          flush=True)

    print(
        f'\tTrain VAE Ploss: {vae_ploss:.3f}', flush=True)
    print(
        f'\tTrain VAE Mloss: {vae_mloss:.3f}', flush=True)
    print(
        f'\tTrain Discr Ploss: {discr_ploss:.3f}', flush=True)
    print(
        f'\tTrain Discr MlossP: {discr_mloss_1:.3f}', flush=True)
    print(
        f'\tTrain Discr MlossN: {discr_mloss_2:.3f}', flush=True)
    print(
        f'\tTrain Discr KLloss: {kl_loss:.3f}', flush=True)
    print(
        f'\t Val. Ploss: {eval_ploss:.3f}', flush=True)
    print(
        f'\t Val. Mloss: {eval_mloss:.3f}', flush=True)
    print(
        f'\t Val. KLloss: {eval_kl_loss:.3f}', flush=True)
    
    
    epoch_loss_writer.add_scalar('train_vae/pitch_loss', vae_ploss, epoch)
    epoch_loss_writer.add_scalar('train_vae/mask_loss', vae_mloss, epoch)
    epoch_loss_writer.add_scalar('train_vae/kl_loss', kl_loss, epoch)
    epoch_loss_writer.add_scalar('train_discr/pitch_loss', discr_ploss, epoch)
    epoch_loss_writer.add_scalar('train_discr/mask_loss_upd_discr', discr_mloss_1, epoch)
    epoch_loss_writer.add_scalar('train_discr/mask_loss_upd_enc', discr_mloss_2, epoch)
    epoch_loss_writer.add_scalar('evaluation/pitch_loss', eval_ploss, epoch)
    epoch_loss_writer.add_scalar('evaluation/mask_loss_upd_discr', eval_mloss, epoch)
    epoch_loss_writer.add_scalar('evaluation/kl_loss', eval_kl_loss, epoch)
    