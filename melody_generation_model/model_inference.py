import pretty_midi as pyd
import torch
import os
import json
import numpy as np
from EC2model import VAE
from util_tools.format_converter import melody_data2matrix, melody_matrix2data, chord_data2matrix, chord_matrix2data



def read_data(directory, name):
    mididir = os.path.join(directory, name+'.mid')
    midi = pyd.PrettyMIDI(mididir)
    melody = melody_data2matrix(midi.instruments[0], midi.get_downbeats())
    chord = chord_data2matrix(midi.instruments[1], midi.get_downbeats(), 'quarter')[:, 12:-12]
    return torch.from_numpy(melody).float().cuda().unsqueeze(0), torch.from_numpy(chord).float().cuda().unsqueeze(0)


def reconstruct(melody, chord):
    dis1, dis2 = model.encoder(melody, chord)
    z1 = dis1.mean
    z2 = dis2.mean
    recon = model.decoder(z1, z2, chord)
    recon = recon.detach().cpu()[0]
    idx = recon.max(1)[1]
    out = torch.zeros_like(recon)
    arange = torch.arange(out.size(0)).long()
    out[arange, idx] = 1
    #print(out.shape, chord.shape)
    mel_recon = melody_matrix2data(out.cpu().detach().numpy())
    chord_recon = chord_matrix2data(chord.cpu().detach().numpy()[0])
    music = pyd.PrettyMIDI()
    music.instruments.append(mel_recon)
    music.instruments.append(chord_recon)
    return music

def control(melody, chord, new_chord):
    dis1, dis2 = model.encoder(melody, chord)
    z1 = dis1.mean
    z2 = dis2.mean
    recon = model.decoder(z1, z2, new_chord)
    recon = recon.detach().cpu()[0]
    idx = recon.max(1)[1]
    out = torch.zeros_like(recon)
    arange = torch.arange(out.size(0)).long()
    out[arange, idx] = 1
    #print(out.shape, chord.shape)
    mel_recon = melody_matrix2data(out.cpu().detach().numpy())
    chord_recon = chord_matrix2data(new_chord.cpu().detach().numpy()[0])
    music = pyd.PrettyMIDI()
    music.instruments.append(mel_recon)
    music.instruments.append(chord_recon)
    return music


with open('./melody_generation_model/model_config.json') as f:
    args = json.load(f)
weight_path = '/gpfsnyu/scratch/jz4807/model-weights/derek/adversarial-ec2vae/params/20210305-14-52-56/best_fitted_params100.pt'
#processor = midi_interface_mono_and_chord()
model = VAE(130, args['Linux_hidden_dim'], 3, 12, args['pitch_dim'], args['rhythm_dim'], args['time_step'])
#model = ensembleModel(130, args['hidden_dim'], 3, 12, args['pitch_dim'], args['rhythm_dim'], args['time_step']).cuda()
params = torch.load(weight_path)
#model.load_state_dict(params['model_state_dict'])
from collections import OrderedDict
renamed_params = OrderedDict()
for k, v in params['model_state_dict'].items():
    name = '.'.join(k.split('.')[1:])
    renamed_params[name] = v
model.load_state_dict(renamed_params)
model.cuda()
#model.load_state_dict(torch.load(weight_path)['model_state_dict'])
model.eval()


ORIGINAL = './melody_generation_model/write/original'
TRANSPOSE = './melody_generation_model/write/transpose+1'
MODAL = './melody_generation_model/write/modal change'
WRITE = './melody_generation_model/melody_write'
if not os.path.exists(WRITE):
    os.makedirs(WRITE)

melody_1, chord_1 = read_data(ORIGINAL, 'M1')
music = reconstruct(melody_1, chord_1)
music.write(os.path.join(WRITE, 'M1_recon.mid'))

melody_2, chord_2 = read_data(ORIGINAL, 'M2')
music = reconstruct(melody_2, chord_2)
music.write(os.path.join(WRITE, 'M2_recon.mid'))

melody_3, chord_3 = read_data(ORIGINAL, 'M3')
music = reconstruct(melody_3, chord_3)
music.write(os.path.join(WRITE, 'M3_recon.mid'))

melody_4, chord_4 = read_data(ORIGINAL, 'M4')
music = reconstruct(melody_4, chord_4)
music.write(os.path.join(WRITE, 'M4_recon.mid'))

melody_5, chord_5 = read_data(ORIGINAL, 'M5')
music = reconstruct(melody_5, chord_5)
music.write(os.path.join(WRITE, 'M5_recon.mid'))



melody_1t, chord_1t = read_data(TRANSPOSE, 'M1')
music = control(melody_1, chord_1, chord_1t)
music.write(os.path.join(WRITE, 'M1_transpose.mid'))

melody_2t, chord_2t = read_data(TRANSPOSE, 'M2')
music = control(melody_2, chord_2, chord_2t)
music.write(os.path.join(WRITE, 'M2_transpose.mid'))

melody_3t, chord_3t = read_data(TRANSPOSE, 'M3')
music = control(melody_3, chord_3, chord_3t)
music.write(os.path.join(WRITE, 'M3_transpose.mid'))

melody_4t, chord_4t = read_data(TRANSPOSE, 'M4')
music = control(melody_4, chord_4, chord_4t)
music.write(os.path.join(WRITE, 'M4_transpose.mid'))

melody_5t, chord_5t = read_data(TRANSPOSE, 'M5')
music = control(melody_5, chord_5, chord_5t)
music.write(os.path.join(WRITE, 'M5_transpose.mid'))



melody_1t, chord_1t = read_data(MODAL, 'M1')
music = control(melody_1, chord_1, chord_1t)
music.write(os.path.join(WRITE, 'M1_modal_change.mid'))

melody_2t, chord_2t = read_data(MODAL, 'M2')
music = control(melody_2, chord_2, chord_2t)
music.write(os.path.join(WRITE, 'M2_modal_change.mid'))

melody_3t, chord_3t = read_data(MODAL, 'M3')
music = control(melody_3, chord_3, chord_3t)
music.write(os.path.join(WRITE, 'M3_modal_change.mid'))

melody_4t, chord_4t = read_data(MODAL, 'M4')
music = control(melody_4, chord_4, chord_4t)
music.write(os.path.join(WRITE, 'M4_modal_change.mid'))

melody_5t, chord_5t = read_data(MODAL, 'M5')
music = control(melody_5, chord_5, chord_5t)
music.write(os.path.join(WRITE, 'M5_modal_change.mid'))
