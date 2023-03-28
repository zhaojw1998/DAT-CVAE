import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.distributions import Normal
from TransformerEncoderLayer import TransformerEncoderLayer
import random
import numpy as np
import pretty_midi as pyd



class VAE(nn.Module):
    def __init__(self, max_simu_note=4+2, max_pitch=11, min_pitch=0,
                 pitch_sos=12, pitch_eos=13, pitch_pad=14, 
                 pitch_hold = 15, pitch_rest =16, pitch_mask = 17,
                 device=None, num_step=32,
                 note_emb_size=128, enc_notes_hid_size=256, enc_time_hid_size=1024, 
                 z_size=128,
                 dec_emb_hid_size=128, dec_time_hid_size=1024, dec_notes_hid_size=512,
                 discr_nhead=8, discr_hid_size=2048, discr_dropout=0.1, discr_nlayer=6
                 ):
        super(VAE, self).__init__()

        # Parameters
        # note and time
        self.max_pitch = max_pitch  # the highest pitch in train/val set.
        self.min_pitch = min_pitch  # the lowest pitch in train/val set.
        self.pitch_sos = pitch_sos
        self.pitch_eos = pitch_eos
        self.pitch_pad = pitch_pad
        self.pitch_hold = pitch_hold
        self.pitch_rest = pitch_rest
        self.pitch_mask = pitch_mask
        self.note_size = (max_pitch - min_pitch + 1) + 3 + 2 + 1 + 10  # including sos, eos, pad, hold, rest, mask, and 10 register indices
        self.max_simu_note = max_simu_note  # the max of notes at each time step.
        self.num_step = num_step  # 32
        self.discr_nlayer = discr_nlayer


        # device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # model_size
        # Used for both encoder & decoder
        self.note_emb_size = note_emb_size
        self.z_size = z_size
        # encoder
        self.enc_notes_hid_size = enc_notes_hid_size
        self.enc_time_hid_size = enc_time_hid_size
        # decoder
        self.dec_emb_hid_size = dec_emb_hid_size
        self.dec_time_hid_size = dec_time_hid_size
        self.dec_init_input = nn.Parameter(torch.rand(2 * self.dec_emb_hid_size))
        self.dec_notes_hid_size = dec_notes_hid_size

        # Modules
        # For both encoder and decoder
        self.enc_note_embedding = nn.Linear(self.note_size, note_emb_size)
        # Encoder
        self.enc_notes_gru = nn.GRU(note_emb_size, enc_notes_hid_size, num_layers=1, batch_first=True, bidirectional=True)
        self.enc_time_gru = nn.GRU(2 * enc_notes_hid_size + note_emb_size, enc_time_hid_size, num_layers=1, batch_first=True, bidirectional=True)
        self.enc_linear_mu = nn.Linear(2 * enc_time_hid_size, z_size)
        self.enc_linear_std = nn.Linear(2 * enc_time_hid_size, z_size)

        # Decoder
        self.z2dec_hid_linear = nn.Linear(z_size, dec_time_hid_size)
        #self.z2dec_in_linear = nn.Linear(z_size, dec_z_in_size)

        self.dec_notes_emb_gru = nn.GRU(note_emb_size, dec_emb_hid_size, num_layers=1, batch_first=True, bidirectional=True)
        self.dec_time_gru = nn.GRU(z_size + note_emb_size + 2 * dec_emb_hid_size, dec_time_hid_size, num_layers=1, batch_first=True, bidirectional=False)
        self.dec_time_to_notes_hid = nn.Linear(dec_time_hid_size, dec_notes_hid_size)
        self.dec_notes_gru = nn.GRU(dec_time_hid_size + note_emb_size, dec_notes_hid_size, num_layers=1, batch_first=True, bidirectional=False)

        self.dec_pitch_out_linear = nn.Linear(dec_notes_hid_size, self.note_size-14) #14D, skip the pad, hold, rest, and mask dimension, preserve sos and eos

        self.discr_Transformer_layers = nn.ModuleDict({})
        for idx in range(discr_nlayer):
            self.discr_Transformer_layers[f'discr_layer_{idx}'] = TransformerEncoderLayer(d_model=note_emb_size*2, nhead=discr_nhead, dim_feedforward=discr_hid_size, dropout=discr_dropout)
            
        self.discr_out_linear = nn.Linear(note_emb_size*2, 12)   #only keep melody pitch dimension (0~127) 


    def get_len_index_tensor(self, ind_x):
        #input ind_x: B * time * simunote * 1
        """Calculate the lengths ((B, 32), torch.LongTensor) of pgrid."""
        with torch.no_grad():
            lengths = self.max_simu_note - \
                      (ind_x[:, :, :, 0] - self.pitch_pad == 0).sum(dim=-1)
        return lengths.cpu()


    def index_tensor_to_multihot_tensor(self, ind_x):
        """Transfer piano_grid to multi-hot piano_grid."""
        # ind_x: (B, 32, max_simu_note, 1)
        with torch.no_grad():
            #dur_part = ind_x[:, :, :, 1:].float()
            out = torch.zeros([ind_x.size(0) * self.num_step * self.max_simu_note,
                               self.note_size],
                              dtype=torch.float).to(self.device)

            out[range(0, out.size(0)), ind_x[:, :, :, 0].view(-1)] = 1.
            out = out.view(-1, self.num_step, self.max_simu_note, self.note_size)
            #out = torch.cat([out[:, :, :, 0: self.note_size], dur_part],
            #                dim=-1)
        return out


    def encoder(self, chord, lengths, melody):
        #chord: (B, num_step, max_simu_note, note_emb_size)
        #lengths: (B, num_step)
        #melody: (B, num_step, note_emb_size)

        chord = chord.view(-1, self.max_simu_note, self.note_emb_size)
        chord = pack_padded_sequence(chord, lengths.view(-1), batch_first=True, enforce_sorted=False)

        chord = self.enc_notes_gru(chord)[-1].transpose(0, 1).contiguous()
        chord = chord.view(-1, self.num_step, 2 * self.enc_notes_hid_size)
        #chord_embedding: (B, 32, 2 * enc_note_hid_size)

        x = torch.cat([chord, melody], dim=-1)
        x = self.enc_time_gru(x)[-1].transpose(0, 1).contiguous()
        # x: (B, 2, enc_time_hid_size)

        x = x.view(x.size(0), -1)
        mu = self.enc_linear_mu(x)  # (B, z_size)
        std = self.enc_linear_std(x).exp_()  # (B, z_size)
        dist = Normal(mu, std)
        return dist, mu


    def get_sos_token(self):
        sos = torch.zeros(self.note_size)
        sos[self.pitch_sos] = 1.
        #sos[self.pitch_range:] = 2.
        sos = sos.to(self.device)
        return sos

    def get_null_chord(self):
        null_chord = torch.zeros(1, self.max_simu_note, self.note_size)
        null_chord[:, 0, self.pitch_sos] = 1.
        null_chord[:, 1, self.pitch_eos] = 1.
        null_chord[:, 2:, self.pitch_pad] = 1.
        null_chord = null_chord.to(self.device)
        null_chord = self.enc_note_embedding(null_chord)
        return null_chord


    def get_hold_embedding(self):
        simu_note = torch.zeors(self.max_simu_note, self.note_size).to(self.device)
        simu_note[0, self.pitch_sos] = 1.
        simu_note[1, self.pitch_eos] = 1.
        simu_note[2:, self.pitch_pad] = 1.
        lengths = torch.LongTensor([1])
        simu_note = pack_padded_sequence(simu_note.unsqueeze(0), lengths.view(-1), batch_first=True,
                                 enforce_sorted=False)
        return simu_note


    def pitch_dur_ind_to_note_token(self, pitch_inds, batch_size):
        token = torch.zeros(batch_size, self.note_size)
        token[range(0, batch_size), pitch_inds] = 1.
        token[token.sum(-1)==0, self.pitch_pad] = 1.
        #token[:, self.pitch_range:] = dur_inds
        token = token.to(self.device)
        token = self.enc_note_embedding(token)
        return token


    def decode_notes(self, notes_summary, melody, batch_size, notes, inference,
                     teacher_forcing_ratio=0.5):
        # notes_summary: (B, 1, dec_time_hid_size)
        # melody: (B, 1, note_emb_size)
        # notes: (B, max_simu_note, note_emb_size), ground_truth
        notes_summary_hid = self.dec_time_to_notes_hid(notes_summary.transpose(0, 1))
        if inference:
            assert teacher_forcing_ratio == 0
            assert notes is None
            sos = self.get_sos_token()  # (note_size,)
            token = self.enc_note_embedding(sos).repeat(batch_size, 1).unsqueeze(1)
            # hid: (B, 1, note_emb_size)
        else:
            token = notes[:, 0].unsqueeze(1)

        predicted_notes = torch.zeros(batch_size, self.max_simu_note,
                                      self.note_emb_size)
        #predicted_notes[:, :, self.pitch_range:] = 2.
        predicted_notes[:, 0] = token.squeeze(1)  # fill sos index
        lengths = torch.zeros(batch_size)
        predicted_notes = predicted_notes.to(self.device)
        #lengths = lengths.to(self.device)
        pitch_outs = []
        #dur_outs = []

        for t in range(1, self.max_simu_note):
            note_summary, notes_summary_hid = \
                self.dec_notes_gru(torch.cat([notes_summary, token], dim=-1),
                                   notes_summary_hid)
            # note_summary: (B, 1, dec_notes_hid_size)
            # notes_summary_hid: (1, B, dec_time_hid_size)

            est_pitch = self.dec_pitch_out_linear(note_summary).squeeze(1)

            pitch_outs.append(est_pitch.unsqueeze(1))
            #dur_outs.append(est_durs.unsqueeze(1))
            pitch_inds = est_pitch.max(1)[1]
            #dur_inds = est_durs.max(2)[1]
            predicted = self.pitch_dur_ind_to_note_token(pitch_inds, batch_size)
            # predicted: (B, note_size)

            predicted_notes[:, t] = predicted
            eos_samp_inds = (pitch_inds == self.pitch_eos).cpu()
            lengths[eos_samp_inds & (lengths == 0)] = t

            if t == self.max_simu_note - 1:
                break
            teacher_force = random.random() < teacher_forcing_ratio
            if inference or not teacher_force:
                token = predicted.unsqueeze(1)
            else:
                token = notes[:, t].unsqueeze(1)
        lengths[lengths == 0] = t
        pitch_outs = torch.cat(pitch_outs, dim=1)
        #dur_outs = torch.cat(dur_outs, dim=1)
        return pitch_outs, predicted_notes, lengths


    def decoder(self, z, melody, inference, x, lengths, teacher_forcing_ratio1,
                teacher_forcing_ratio2):
        # z: (B, z_size)
        # melody: (B, num_step, note_emb_size)
        # x: (B, num_step, max_simu_note, note_emb_size)
        batch_size = z.size(0)
        z_hid = self.z2dec_hid_linear(z).unsqueeze(0)
        # z_hid: (1, B, dec_time_hid_size)
        z_in = z.unsqueeze(1)
        #z_in = self.z2dec_in_linear(z).unsqueeze(1)
        # z_in: (B, z_size)

        if inference:
            assert x is None
            assert lengths is None
            assert teacher_forcing_ratio1 == 0
            assert teacher_forcing_ratio2 == 0
        else:
            x_summarized = x.view(-1, self.max_simu_note, self.note_emb_size)
            x_summarized = pack_padded_sequence(x_summarized, lengths.view(-1),
                                                batch_first=True,
                                                enforce_sorted=False)
            x_summarized = self.dec_notes_emb_gru(x_summarized)[-1].transpose(0, 1).contiguous()
            x_summarized = x_summarized.view(-1, self.num_step, 2 * self.dec_emb_hid_size)

        pitch_outs = []
        #dur_outs = []
        token = self.dec_init_input.repeat(batch_size, 1).unsqueeze(1)
        # (B, 2 * dec_emb_hid_size)

        #init_notes = self.get_null_chord().repeat(batch_size, 1, 1)
        #init_length = torch.ones(batch_size) * 2
        #token = pack_padded_sequence(init_notes,
        #                            init_length,
        #                            batch_first=True,
        #                            enforce_sorted=False)
        #token = self.dec_notes_emb_gru(token)[-1].transpose(0, 1).contiguous()
        #token = token.view(-1, 2 * self.dec_emb_hid_size).unsqueeze(1)

        for t in range(self.num_step):
            notes_summary, z_hid = \
                self.dec_time_gru(torch.cat([z_in, melody[:, t].unsqueeze(1), token], dim=-1), z_hid)
            if inference:
                pitch_out, predicted_notes, predicted_lengths = \
                    self.decode_notes(notes_summary, melody[:, t].unsqueeze(1), batch_size, None,
                                      inference, teacher_forcing_ratio2)
            else:
                pitch_out, predicted_notes, predicted_lengths = \
                    self.decode_notes(notes_summary, melody[:, t].unsqueeze(1), batch_size, x[:, t],
                                      inference, teacher_forcing_ratio2)
            pitch_outs.append(pitch_out.unsqueeze(1))
            #dur_outs.append(dur_out.unsqueeze(1))
            if t == self.num_step - 1:
                break

            teacher_force = random.random() < teacher_forcing_ratio1
            if teacher_force and not inference:
                token = x_summarized[:, t].unsqueeze(1)
            else:
                token = pack_padded_sequence(predicted_notes,
                                             predicted_lengths,
                                             batch_first=True,
                                             enforce_sorted=False)
                token = self.dec_notes_emb_gru(token)[-1].\
                    transpose(0, 1).contiguous()
                token = token.view(-1, 2 * self.dec_emb_hid_size).unsqueeze(1)
        pitch_outs = torch.cat(pitch_outs, dim=1)
        #dur_outs = torch.cat(dur_outs, dim=1)
        # print(pitch_outs.size())
        # print(dur_outs.size())
        return pitch_outs#, dur_outs

   
    def discr(self, mu, x, masked_idx):
        #mu: (B, note_embed_size)
        #x: (B, num_step*4, note_embed_size), masked melody
        #masked_idx: [[B, B, B, B, ...], [T, T, T, T, ...]]
        x = torch.cat([x, mu.unsqueeze(1).repeat(1, x.shape[1], 1)], axis=-1)

        for idx in range(self.discr_nlayer):
            x = self.discr_Transformer_layers[f'discr_layer_{idx}'](x)

        x = self.discr_out_linear(x)  #(B, num_step*4, 128)

        return x[masked_idx[0], masked_idx[1], :]



    def forward(self, chord, melody, masked_melody, inference=False, sample=True,
                teacher_forcing_ratio1=0.5, teacher_forcing_ratio2=0.5):
        # chord: (batch, num_step, max_simu_note, 1)
        # melody: (batch, num_step*4, 28)
        # masked_melody: (batch, num_step*4, 28), 28 = chroma(0~11) + Vacant(12~14) + hold(15) + rest(16) + mask(17) + register(10)

        lengths = self.get_len_index_tensor(chord)  # lengths: (batch, num_step)
        chord = self.index_tensor_to_multihot_tensor(chord)
        chord = self.enc_note_embedding(chord)    #(batch, num_step, max_simu_note, note_emb_size)
        
        melody = self.enc_note_embedding(melody)    #(batch, num_step*4, note_emb_size)
        #melody = self.positional_embedding(melody, 'quaver')

        masked_idx = torch.nonzero(masked_melody[:, :, self.pitch_mask], as_tuple=True)
        masked_melody[:, :, self.pitch_mask] = 0.
        masked_melody = self.enc_note_embedding(masked_melody)    #(batch, num_step*4, note_emb_size)
        
        melody_beat_summary = melody[:, ::4, :] + melody[:, 1::4, :] + melody[:, 2::4, :] + melody[:, 3::4, :]
        dist, mu, = self.encoder(chord, lengths, melody_beat_summary)

        if sample:
            z = dist.rsample()
        else:
            z = dist.mean

        if inference:
            pitch_outs = self.decoder(z, melody_beat_summary, 
                                    inference, x = None, lengths = None, 
                                    teacher_forcing_ratio1 = 0., teacher_forcing_ratio2 = 0.)
        else:                        
            pitch_outs = self.decoder(z, melody_beat_summary, 
                                    inference, chord, lengths, 
                                    teacher_forcing_ratio1, teacher_forcing_ratio2)
        
        mask_prediction = self.discr(mu, masked_melody, masked_idx)

        return pitch_outs, dist, mask_prediction, masked_idx


    def grid_to_pr_and_notes(self, est_pitch, bpm=60., start=0.):
        try:
            est_pitch = est_pitch.max(-1)[1].cpu().detach().numpy() #(32, max_simu_note-1), NO BATCH HERE
        except:
            pass
        if est_pitch.shape[1] == self.max_simu_note:
            est_pitch = est_pitch[:, 1:]
        
        harmonic_rhythm = 1. - (est_pitch[:, 0]==self.pitch_eos)*1.

        pr = np.zeros((32, 128), dtype=int)
        alpha = 0.25 * 60 / bpm
        notes = []
        for t in range(self.num_step):
            for n in range(self.max_simu_note-1):
                note = est_pitch[t, n]
                if note == self.pitch_eos:
                    break
                pitch = note + self.min_pitch
                duration = 1
                for j in range(t+1, self.num_step):
                    if harmonic_rhythm[j] == 1:
                        break
                    duration +=1
                pr[t, pitch] = min(duration, 32 - t)
                notes.append(
                    pyd.Note(100, int(pitch), start + t * alpha, start + (t + duration) * alpha))
        return pr, notes


    def encoder_params(self):
        for name, param in self.named_parameters(recurse=True):
            if 'enc' in name:
                yield param
    

    def decoder_params(self):
        for name, param in self.named_parameters(recurse=True):
            if 'dec' in name:
                yield param


    def discr_params(self):
        for name, param in self.named_parameters(recurse=True):
            if 'discr' in name:
                yield param


if __name__ == '__main__':
    
    #import sys
    #sys.path.append('AD-PianoTree-VAE-Alt8')
    import utils
    config_fn = './model_config.json'
    #train_hyperparams = utils.load_params_dict('train_hyperparams', config_fn)
    model_params = utils.load_params_dict('model_params', config_fn)
    data_repr_params = utils.load_params_dict('data_repr', config_fn)
    #project_params = utils.load_params_dict('project', config_fn)
    #dataset_path = utils.load_params_dict('dataset_paths', config_fn)

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
    #for parameter in model.parameters():
    #    print(parameter.shape)
    #sys.exit()
    
    #weight_path = './PianoTree-VAE/ptree_model.pt'
    #params = torch.load(weight_path)
    #if 'model_state_dict' in params:
    #    params = params['model_state_dict']
    #model.load_state_dict(params)
    model.cuda()
    model.eval()

    for name, param in model.named_parameters(recurse=True):
        print(name, param.shape)


    from nottingham_dataset import Nottingham
    from torch.utils.data import DataLoader
    dataset = np.load('./code/data.npy', allow_pickle=True)
    dataset = Nottingham(dataset=dataset, 
                        length=128, 
                        step_size=16, 
                        chord_fomat='pr')
    data_loader = DataLoader(dataset, batch_size=4, shuffle=False)
    print(len(data_loader))
    for i, (chord, melody_gt, melody_28D, masked_melody) in enumerate(data_loader):
        chord = chord.long().cuda() # this is an indice tensor
        melody_gt = melody_gt.float().cuda()  # this is a one-hot tensor
        melody_28D = melody_28D.float().cuda()  # this is a multi-hot tensor
        masked_melody = masked_melody.float().cuda()  # this is a multi-hot tensor
        print(chord.shape, melody_gt.shape, melody_28D.shape, masked_melody.shape)
        pitch_outs, dist, mask_output, mask_idx = model(chord, melody_28D, masked_melody, inference=True, sample=None, teacher_forcing_ratio1=0, teacher_forcing_ratio2=0)
        melody_gt = melody_gt[mask_idx[0], mask_idx[1], :128]
        print(pitch_outs.shape, mask_output.shape, melody_gt.shape)
        print(mask_output.max(-1)[-1])
        print(melody_gt.max(-1)[-1])
 
        #pr, notes = model.grid_to_pr_and_notes(pitch_outs[0], bpm=30., start=0.)
        #print(pr.shape)
        break