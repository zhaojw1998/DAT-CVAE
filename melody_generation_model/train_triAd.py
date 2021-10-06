import sys
sys.path.append('./melody_generation_model')
import json
import torch
torch.cuda.current_device()
import os
import numpy as np
from torch import optim
from torch.distributions import kl_divergence, Normal
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import time
from collections import OrderedDict
import platform
from vae_triAd import ensembleModel
from ruihan_data_loader import MusicArrayLoader_ruihan

class MinExponentialLR(ExponentialLR):
    def __init__(self, optimizer, gamma, minimum, last_epoch=-1):
        self.min = minimum
        super(MinExponentialLR, self).__init__(optimizer, gamma, last_epoch=last_epoch)

    def get_lr(self):
        return [
            max(base_lr * self.gamma**self.last_epoch, self.min)
            for base_lr in self.base_lrs
        ]

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def std_normal(shape):
    N = Normal(torch.zeros(shape), torch.ones(shape))
    if torch.cuda.is_available():
        N.loc = N.loc.cuda()
        N.scale = N.scale.cuda()
    return N

def loss_function_vae(recon,
                  recon_rhythm,
                  target_tensor,
                  rhythm_target,
                  distribution_1,
                  distribution_2,
                  beta=.1):
    CE1 = F.nll_loss(
        recon.view(-1, recon.size(-1)),
        target_tensor,
        reduction='mean')
    #print('melody_recon loss:', CE1)
    CE2 = F.nll_loss(
        recon_rhythm.view(-1, recon_rhythm.size(-1)),
        rhythm_target,
        reduction='mean')
    #print('rhythm_recon loss:', CE2)
    normal1 = std_normal(distribution_1.mean.size())
    normal2 = std_normal(distribution_2.mean.size())
    KLD1 = kl_divergence(distribution_1, normal1).mean()
    KLD2 = kl_divergence(distribution_2, normal2).mean()
    return CE1 + CE2 + beta * (KLD1 + KLD2), CE1 + CE2, KLD1 + KLD2

def loss_function_discr(chord_prediction, target_chord, distribution_1, distribution_2, beta=.1):
    chord_prediction.view(-1, chord_prediction.size(-1))
    target_chord.view(-1, target_chord.size(-1))
    criterion = torch.nn.BCELoss(weight=None, reduction='mean')
    CE_chord = criterion(chord_prediction, target_chord)
    normal1 = std_normal(distribution_1.mean.size())
    normal2 = std_normal(distribution_2.mean.size())
    KLD1 = kl_divergence(distribution_1, normal1).mean()
    KLD2 = kl_divergence(distribution_2, normal2).mean()
    return CE_chord + beta * (KLD1 + KLD2), CE_chord, KLD1 + KLD2

def chord_shift(chord, lb, hb):
    #print(chord.shape)
    shifted_chord =[]
    for i in range(chord.shape[0]):
        shift = np.random.randint(lb, hb+1)
        shifted_chord.append(torch.roll(chord[i], shift, dims=-1))
    out = torch.stack(shifted_chord, 0)
    #print(out.shape)
    return out

def train_vae_for_ruihan(step, step_whole, model, train_dataloader, batch_size, loss_function_vae, loss_function_discr, optimizer, scheduler_vae, scheduler_discr, scheduler_enc, writer, args, cycle_consistency=False, beta=0.1):
    batch, c = train_dataloader.get_batch(batch_size)    #batch : batch * 32 *142; c: batch * 32 * 12 
    encode_tensor = torch.from_numpy(batch).float()
    c = torch.from_numpy(c).float()
    rhythm_target = np.expand_dims(batch[:, :, :-2].sum(-1), -1)    #batch* n_step* 1
    rhythm_target = np.concatenate((rhythm_target, batch[:, :, -2:]), -1)   #batch* n_step* 3
    rhythm_target = torch.from_numpy(rhythm_target).float()
    rhythm_target = rhythm_target.view(-1, rhythm_target.size(-1)).max(-1)[1]
    target_tensor = encode_tensor.view(-1, encode_tensor.size(-1)).max(-1)[1]
    if torch.cuda.is_available():
        encode_tensor = encode_tensor.cuda()
        target_tensor = target_tensor.cuda()
        rhythm_target = rhythm_target.cuda()
        c = c.cuda()
    optimizer.zero_grad()
    
    (recon, recon_rhythm, dis1m, dis1s, dis2m, dis2s), chord_prediction = model(encode_tensor, c)
    
    distribution_1 = Normal(dis1m, dis1s)
    distribution_2 = Normal(dis2m, dis2s)
    loss, l_recon, l_kl = loss_function_vae(recon, recon_rhythm, target_tensor, rhythm_target, distribution_1, distribution_2, beta=args['beta'])
    loss_chord_KL, loss_chord, loss_KL = loss_function_discr(chord_prediction, c, distribution_1, distribution_2)   #theoretical optimal value is ln2
    #loss = loss + beta*loss_chord
    losses_recon1.update(l_recon.item())
    losses_kl1.update(l_kl.item())
    losses_chord1.update(loss_chord.item())

    if cycle_consistency:
        c_shift = chord_shift(c, -2, 2)
        #print('melody:', encode_tensor)
        #print('chord:', c_shift)
        (recon_s, _, _, _, _, _), chord_prediction = model(encode_tensor, c, c_shift)
        #print('recon_melody:', F.gumbel_softmax(recon_s, tau=1, hard=True, dim=-1)) 
        #print('recon_chord:', chord_prediction)
        (recon_s, recon_rhythm_s, dis1m_s, dis1s_s, dis2m_s, dis2s_s), _ = model(F.gumbel_softmax(recon_s, tau=1, hard=True, dim=-1), c_shift, c)
        distribution_1_s = Normal(dis1m_s, dis1s_s)
        distribution_2_s = Normal(dis2m_s, dis2s_s)
        loss_s, l_recon_s, l_kl_s = loss_function_vae(recon_s, recon_rhythm_s, target_tensor, rhythm_target, distribution_1_s, distribution_2_s, beta=args['beta'])
        losses1_s.update(loss_s.item())
        losses_recon1_s.update(l_recon_s.item())
        losses_kl1_s.update(l_kl_s.item())
        print('non-cycle loss:', loss)
        loss = loss + loss_s
        print('cycle consistency loss:', loss_s)
        
    
    losses1.update(loss.item())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.vae.parameters(), 1)
    optimizer.step()


    print('---------------------------Training VAE----------------------------')
    for param in optimizer.param_groups:
        print('lr1: ', param['lr'])
    num_batch = train_dataloader.get_n_sample() // batch_size
    print('Epoch: [{0}][{1}/{2}]'.format(train_dataloader.get_n_epoch(), (step)%num_batch, num_batch))
    print('loss: {loss:.5f}'.format(loss=losses1.avg))
    writer.add_scalar('train_vae/1-loss_plain-epoch', (losses1.avg - losses1_s.avg), step)
    writer.add_scalar('train_vae/2-loss_recon-epoch', losses_recon1.avg, step)
    writer.add_scalar('train_vae/3-loss_KL-epoch', losses_kl1.avg, step)
    if cycle_consistency:
        writer.add_scalar('train_vae/4-loss_S_total-epoch', losses1_s.avg, step)
        writer.add_scalar('train_vae/5-loss_S_recon-epoch', losses_recon1_s.avg, step)
        writer.add_scalar('train_vae/6-loss_S_KL-epoch', losses_kl1_s.avg, step)
    writer.add_scalar('train_vae/7-loss_chord-epoch', losses_chord1.avg, step)
    writer.add_scalar('train_vae/8-learning-rate', param['lr'], step)

    step += 1
    step_whole += 1
    if args['decay'] > 0:
        scheduler_vae.step()
        #scheduler_discr.step()
        #scheduler_enc.step()
    #train_dataloader.shuffle_samples()
    return step, step_whole

def train_discr_for_ruihan(step, step_whole, model, train_dataloader, batch_size, loss_function_vae, loss_function_discr, optimizer_discr, optimizer_enc, scheduler_vae, scheduler_discr, scheduler_enc, writer, args, beta=0.1):
    batch, c = train_dataloader.get_batch(batch_size)    #batch : batch * 32 *142; c: batch * 32 * 12 
    encode_tensor = torch.from_numpy(batch).float()
    c = torch.from_numpy(c).float()
    rhythm_target = np.expand_dims(batch[:, :, :-2].sum(-1), -1)    #batch* n_step* 1
    rhythm_target = np.concatenate((rhythm_target, batch[:, :, -2:]), -1)   #batch* n_step* 3
    rhythm_target = torch.from_numpy(rhythm_target).float()
    rhythm_target = rhythm_target.view(-1, rhythm_target.size(-1)).max(-1)[1]
    target_tensor = encode_tensor.view(-1, encode_tensor.size(-1)).max(-1)[1]
    if torch.cuda.is_available():
        encode_tensor = encode_tensor.cuda()
        target_tensor = target_tensor.cuda()
        rhythm_target = rhythm_target.cuda()
        c = c.cuda()

    epoch = train_dataloader.get_n_epoch()
    thresh = max(3 - int(epoch / 10 * 3), 0)
    if step % 5 <= thresh:
        target = 'Discr'
        optimizer = optimizer_discr
        optimizer.zero_grad()
        (recon, recon_rhythm, dis1m, dis1s, dis2m, dis2s), chord_prediction = model(encode_tensor, c)
        distribution_1 = Normal(dis1m, dis1s)
        distribution_2 = Normal(dis2m, dis2s)
        loss_chord_KL, loss_chord, loss_KL = loss_function_discr(chord_prediction, c, distribution_1, distribution_2)
        with torch.no_grad():
            loss, l_recon, l_kl = loss_function_vae(recon, recon_rhythm, target_tensor, rhythm_target, distribution_1, distribution_2, beta=args['beta'])
        loss_chord_KL.backward()
        losses2.update(loss.item())
        losses_recon2.update(l_recon.item())
        losses_kl2.update(l_kl.item())
        losses_chord2_1.update(loss_chord.item())
        torch.nn.utils.clip_grad_norm_(model.discr.parameters(), 1)
        optimizer.step()
    else:
        target = 'Enc'
        optimizer = optimizer_enc
        optimizer.zero_grad()
        (recon, recon_rhythm, dis1m, dis1s, dis2m, dis2s), chord_prediction = model(encode_tensor, c)
        distribution_1 = Normal(dis1m, dis1s)
        distribution_2 = Normal(dis2m, dis2s)
        loss_chord_KL, loss_chord, loss_KL = loss_function_discr(chord_prediction, 1-c, distribution_1, distribution_2)
        with torch.no_grad():
            loss, l_recon, l_kl = loss_function_vae(recon, recon_rhythm, target_tensor, rhythm_target, distribution_1, distribution_2, beta=args['beta'])
        loss_chord_KL.backward()
        losses2.update(loss.item())
        losses_recon2.update(l_recon.item())
        losses_kl2.update(l_kl.item())
        losses_chord2_2.update(loss_chord.item())
        torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1)
        optimizer.step()

    print('---------------------------Training ' + target + ' ----------------------------')
    for param in optimizer.param_groups:
        print('lr1: ', param['lr'])
    num_batch = train_dataloader.get_n_sample() // batch_size
    print('Epoch: [{0}][{1}/{2}]'.format(train_dataloader.get_n_epoch(), (step)%num_batch, num_batch))
    print('loss: {loss:.5f}'.format(loss=losses2.avg))
    writer.add_scalar('train_discr/1-loss_total-epoch', losses2.avg, step)
    writer.add_scalar('train_discr/2-loss_recon-epoch', losses_recon2.avg, step)
    writer.add_scalar('train_discr/3-loss_KL-epoch', losses_kl2.avg, step)
    if target == 'Discr':
        writer.add_scalar('train_discr/4-Discr_loss_chord-epoch', losses_chord2_1.avg, step)
    elif target == 'Enc':
        writer.add_scalar('train_discr/4-Enc_loss_chord-epoch', losses_chord2_2.avg, step)
    writer.add_scalar('train_discr/5-learning-rate', param['lr'], step)

    step += 1
    step_whole += 1
    if args['decay'] > 0:
        #scheduler_vae.step()
        scheduler_discr.step()
        scheduler_enc.step()
    #train_dataloader.shuffle_samples()
    return step, step_whole

def validation_for_ruihan(step, epoch, model, val_dataloader, batch_size, loss_function_vae, loss_function_discr, writer, args, cycle_consistency=False, beta=0.1):
    with torch.no_grad():
        batch, c = val_dataloader.get_batch(batch_size)    #batch : batch * 32 *142; c: batch * 32 * 12 
        encode_tensor = torch.from_numpy(batch).float()
        c = torch.from_numpy(c).float()
        rhythm_target = np.expand_dims(batch[:, :, :-2].sum(-1), -1)    #batch* n_step* 1
        rhythm_target = np.concatenate((rhythm_target, batch[:, :, -2:]), -1)   #batch* n_step* 3
        rhythm_target = torch.from_numpy(rhythm_target).float()
        rhythm_target = rhythm_target.view(-1, rhythm_target.size(-1)).max(-1)[1]
        target_tensor = encode_tensor.view(-1, encode_tensor.size(-1)).max(-1)[1]
        if torch.cuda.is_available():
            encode_tensor = encode_tensor.cuda()
            target_tensor = target_tensor.cuda()
            rhythm_target = rhythm_target.cuda()
            c = c.cuda()
        (recon, recon_rhythm, dis1m, dis1s, dis2m, dis2s), chord_prediction = model(encode_tensor, c)
        distribution_1 = Normal(dis1m, dis1s)
        distribution_2 = Normal(dis2m, dis2s)
        loss, l_recon, l_kl = loss_function_vae(recon, recon_rhythm, target_tensor, rhythm_target, distribution_1, distribution_2, beta=args['beta'])
        loss_chord_KL, loss_chord, loss_KL = loss_function_discr(chord_prediction, c, distribution_1, distribution_2)
        #loss = loss + beta*loss_chord
        losses_recon3.update(l_recon.item())
        losses_kl3.update(l_kl.item())
        losses_chord3.update(loss_chord.item())

        if cycle_consistency:
            c_shift = chord_shift(c, -2, 2)
            (recon_s, recon_rhythm_s, dis1m_s, dis1s_s, dis2m_s, dis2s_s), chord_prediction_s = model(encode_tensor, c, c_shift)
            (recon_s, recon_rhythm_s, dis1m_s, dis1s_s, dis2m_s, dis2s_s), chord_prediction_s = model(recon_s, c_shift, c)
            distribution_1_s = Normal(dis1m_s, dis1s_s)
            distribution_2_s = Normal(dis2m_s, dis2s_s)
            loss_s, l_recon_s, l_kl_s = loss_function_vae(recon_s, recon_rhythm_s, target_tensor, rhythm_target, distribution_1_s, distribution_2_s, beta=args['beta'])
            losses3_s.update(loss_s.item())
            losses_recon3_s.update(l_recon_s.item())
            losses_kl3_s.update(l_kl_s.item())
            loss = loss + loss_s
        losses3.update(loss.item())
            
    print('----validation----')
    num_batch = val_dataloader.get_n_sample() // batch_size
    print('Epoch: [{0}][{1}/{2}]'.format(epoch+1, step, num_batch))
    print('loss: {loss:.5f}'.format(loss=losses3.avg))

    writer.add_scalar('val/1-loss_total-epoch', losses3.avg, pre_epoch+1)
    writer.add_scalar('val/2-loss_recon-epoch', losses_recon3.avg, pre_epoch+1)
    writer.add_scalar('val/3-loss_KL-epoch', losses_kl3.avg, pre_epoch+1)
    writer.add_scalar('val/4-loss_S_total-epoch', losses3_s.avg, pre_epoch+1)
    writer.add_scalar('val/5-loss_S_recon-epoch', losses_recon3_s.avg, pre_epoch+1)
    writer.add_scalar('val/6-loss_S_KL-epoch', losses_kl3_s.avg, pre_epoch+1)
    writer.add_scalar('val/7-loss_chord-epoch', losses_chord3.avg, pre_epoch+1)
    
    step += 1
    #val_dataloader.shuffle_samples()
    return step, losses_recon3.avg


# some initialization
debug_mode = (platform.system() == 'Windows')
adversarail_mode = True
cycle_consistency_mode = False

with open('./melody_generation_model/model_config.json') as f:
    args = json.load(f)
run_time = time.strftime("%Y%m%d-%H-%M-%S", time.localtime())
logdir = run_time + '/log'
save_dir = run_time + '/model'
if debug_mode == False:
    logdir = os.path.join(args["Linux_log_save"], logdir)
    save_dir = os.path.join(args["Linux_log_save"], save_dir)
    batch_size = args['Linux_batch_size']
    augment = False
    hidden_dim = args['Linux_hidden_dim']
    data_path = args['Linux_data_path']
else:
    logdir = os.path.join(args["log_save"], logdir)
    save_dir = os.path.join(args["log_save"], save_dir)
    batch_size = args['batch_size']
    augment = False
    hidden_dim = args['hidden_dim']
    data_path = args['data_path']
if not os.path.exists(logdir):
    os.makedirs(logdir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)



model = ensembleModel(130, hidden_dim, 3, 12, args['pitch_dim'], args['rhythm_dim'], args['time_step'], k=1000, latent_output='mu')
writer = SummaryWriter(logdir)
optimizer_vae = optim.Adam(model.vae.parameters(), lr=args['lr'])
optimizer_discr = optim.Adam(model.discr.parameters(), lr=args['lr'])
optimizer_enc = optim.Adam(model.encoder.parameters(), lr=args['lr'])
optimizer_full = optim.Adam(model.parameters(), lr=args['lr'])
#if args['decay'] > 0:
scheduler_vae = MinExponentialLR(optimizer_vae, gamma=args['decay'], minimum=1e-5,)
scheduler_discr = MinExponentialLR(optimizer_discr, gamma=args['decay_discr'], minimum=1e-5,)
scheduler_enc = MinExponentialLR(optimizer_enc, gamma=args['decay_discr'], minimum=1e-5,)
#schedular_full = MinExponentialLR(optimizer_full, gamma=args['decay'], minimum=1e-5,)
if torch.cuda.is_available():
    print('Using: ', torch.cuda.get_device_name(torch.cuda.current_device()))
    model.cuda()
else:
    print('CPU mode')
# end of initialization

#model = torch.nn.DataParallel(model, device_ids=[0, 1])

dataset = np.load(data_path, allow_pickle=True).T
np.random.seed(0)
np.random.shuffle(dataset)
if debug_mode:
    anchor = int(dataset.shape[0] * 0.005)
    train_data = dataset[:anchor, :]
    #train_data = dataset[anchor:, :]
    anchor = int(dataset.shape[0] * 0.995)
    val_data = dataset[anchor:, :]
else:
    anchor = int(dataset.shape[0] * 0.95)
    train_data = dataset[:anchor, :]
    val_data = dataset[anchor:, :]
train_data = train_data.T
val_data = val_data.T
dl_train = MusicArrayLoader_ruihan(train_data, 32, 16, augment)
dl_train.chunking()
dl_train_discr = MusicArrayLoader_ruihan(train_data, 32, 16, augment)
dl_train_discr.chunking()
dl_val = MusicArrayLoader_ruihan(val_data, 32, 16, augment)
dl_val.chunking()

dl_train.shuffle_samples()
dl_val.shuffle_samples()
dl_train_discr.shuffle_samples()
print('train volumn:', dl_train.get_n_sample(), 'val volumn:', dl_val.get_n_sample())

step_vae = 0
step_discr = 0
step_whole = 0
pre_epoch = 0
epoch_val = 0
val_loss_record = 100
#recorder for training vae
losses1 = AverageMeter()
losses_recon1 = AverageMeter()
losses_kl1 = AverageMeter()
losses1_s = AverageMeter()
losses_recon1_s = AverageMeter()
losses_kl1_s = AverageMeter()
losses_chord1 = AverageMeter()
#recorder for training discriminator
losses2 = AverageMeter()
losses_recon2 = AverageMeter()
losses_kl2 = AverageMeter()
losses_chord2_1 = AverageMeter()
losses_chord2_2 = AverageMeter()
#recorder for validation
losses3 = AverageMeter()
losses_recon3 = AverageMeter()
losses_kl3 = AverageMeter()
losses3_s = AverageMeter()
losses_recon3_s = AverageMeter()
losses_kl3_s = AverageMeter()
losses_chord3 = AverageMeter()

while dl_train.get_n_epoch() < args['n_epochs']:
    if dl_train.get_n_epoch() != pre_epoch:
        step_val = 0
        model.eval()
        while dl_val.get_n_epoch() == epoch_val:
            #try:
            step_val, loss_output = validation_for_ruihan(step_val, pre_epoch, model, dl_val, batch_size, loss_function_vae, loss_function_discr, writer, args, cycle_consistency=cycle_consistency_mode)
        pre_epoch = dl_train.get_n_epoch()
        epoch_val = dl_val.get_n_epoch()
        dl_train.shuffle_samples()
        dl_val.shuffle_samples()
        dl_train_discr.shuffle_samples()
        #if (pre_epoch + 1) % 10 == 0:
        #if loss_output < val_loss_record:
        val_loss_record = loss_output
        checkpoint = save_dir + '/best_fitted_params' + str(pre_epoch).zfill(3) + '.pt'
        torch.save({'epoch': pre_epoch, 'model_state_dict': model.vae.cpu().state_dict(), 'model_full_state_dict': model.cpu().state_dict(), 'optimizer_state_dict': optimizer_full.state_dict()}, checkpoint)
        model.cuda()
        print('Model saved!')
    
    model.train()
    if adversarail_mode == False:
        step_vae, step_whole = train_vae_for_ruihan(step_vae, step_whole, model, dl_train, batch_size, loss_function_vae, loss_function_discr, optimizer_vae, scheduler_vae, scheduler_discr, scheduler_enc, writer, args, cycle_consistency=cycle_consistency_mode)
    else:
        for i in range(100): #train discriminator %x should be modified as well
            #try:
            step_vae, step_whole = train_vae_for_ruihan(step_vae, step_whole, model, dl_train, batch_size, loss_function_vae, loss_function_discr, optimizer_vae, scheduler_vae, scheduler_discr, scheduler_enc, writer, args, cycle_consistency=cycle_consistency_mode)
            
        for i in range(5):
            #try:
            step_discr, step_whole = train_discr_for_ruihan(step_discr, step_whole, model, dl_train_discr, batch_size, loss_function_vae, loss_function_discr, optimizer_discr, optimizer_enc, scheduler_vae, scheduler_discr, scheduler_enc, writer, args)

