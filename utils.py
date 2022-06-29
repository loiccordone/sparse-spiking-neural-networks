import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.io.wavfile as wav
import torch
from datetime import datetime

import struct
import itertools

def txt2list(filename):
    lines_list = []
    with open(filename, 'r') as txt:
        for line in txt:
            lines_list.append(line.rstrip('\n'))
    return lines_list

def plot_spk_rec(spk_rec, idx):
    

    nb_plt = len(idx)
    d = int(np.sqrt(nb_plt))
    gs = GridSpec(d,d)
    fig= plt.figure(figsize=(30,20),dpi=150)
    for i in range(nb_plt):
        plt.subplot(gs[i])
        plt.imshow(spk_rec[idx[i]].T,cmap=plt.cm.gray_r, origin="lower", aspect='auto')
        if i==0:
            plt.xlabel("Time")
            plt.ylabel("Units")    
    
    
def plot_mem_rec(mem, idx):
    
    nb_plt = len(idx)
    d = int(np.sqrt(nb_plt))
    dim = (d, d)
    
    gs=GridSpec(*dim)
    plt.figure(figsize=(30,20))
    dat = mem[idx]
        
    for i in range(nb_plt):
        if i==0: a0=ax=plt.subplot(gs[i])
        else: ax=plt.subplot(gs[i],sharey=a0)
        ax.plot(dat[i])
        
def get_random_noise(noise_files, size):
    
    noise_idx = np.random.choice(len(noise_files))
    fs, noise_wav = wav.read(noise_files[noise_idx])
    
    offset = np.random.randint(len(noise_wav)-size)
    noise_wav = noise_wav[offset:offset+size].astype(float)
    
    return noise_wav

def generate_random_silence_files(nb_files, noise_files, size, prefix, sr=16000):
    
    for i in range(nb_files):
        
        silence_wav = get_random_noise(noise_files, size)
        wav.write(prefix+"_"+str(i)+".wav", sr, silence_wav)
        
        
def split_wav(waveform, frame_size, split_hop_length):
    
    splitted_wav = []
    offset = 0
    
    while offset + frame_size < len(waveform):
        splitted_wav.append(waveform[offset:offset+frame_size])
        offset += split_hop_length
        
    return splitted_wav

def read_events(file_read, x_dim, y_dim):
    """A simple function that reads events from cAER tcp.

    Args:
        file_read (TYPE): Description
        xdim (TYPE): Description
        ydim (TYPE): Description

    Returns:
        TYPE: Description
    """

    # raise Exception
    data = file_read.read(28)

    if (len(data) == 0):
        return [-1], [-1], [-1], [-1], [-1], [-1]

    # read header
    eventtype = struct.unpack('H', data[0:2])[0]
    eventsource = struct.unpack('H', data[2:4])[0]
    eventsize = struct.unpack('I', data[4:8])[0]
    eventoffset = struct.unpack('I', data[8:12])[0]
    eventtsoverflow = struct.unpack('I', data[12:16])[0]
    eventcapacity = struct.unpack('I', data[16:20])[0]
    eventnumber = struct.unpack('I', data[20:24])[0]
    eventvalid = struct.unpack('I', data[24:28])[0]
    next_read = eventcapacity * eventsize  # we now read the full packet
    data = file_read.read(next_read)
    counter = 0  # eventnumber[0]
    # return arrays
    x_addr_tot = []
    y_addr_tot = []
    pol_tot = []
    ts_tot = []
    spec_type_tot = []
    spec_ts_tot = []

    if (eventtype == 1):  
        while (data[counter:counter + eventsize]):  # loop over all event packets
            aer_data = struct.unpack('I', data[counter:counter + 4])[0]
            timestamp = struct.unpack('I', data[counter + 4:counter + 8])[0]
            
            x_addr_tot.append((aer_data >> 17) & 0x00007FFF)
            y_addr_tot.append((aer_data >> 2) & 0x00007FFF)
            pol_tot.append((aer_data >> 1) & 0x00000001)
            ts_tot.append(timestamp)
            
            counter = counter + eventsize

    return (np.array(x_addr_tot), np.array(y_addr_tot), np.array(pol_tot), np.array(ts_tot), np.array(spec_type_tot),
            np.array(spec_ts_tot))

def aedat2torch(datafile):
    with open(datafile, 'rb') as aerfile:

        # Skip the header
        while aerfile.readline() != b'#!END-HEADER\r\n':
            continue

        X_DIM = 128
        Y_DIM = 128

        ts_events_tmp = []
        x_events_tmp = []
        y_events_tmp = []
        p_events_tmp = []
        
        while (1):
            x, y, p, ts_tot, spec_type, spec_type_ts = read_events(aerfile, X_DIM, Y_DIM)
            if (len(ts_tot) > 0 and ts_tot[0] == -1): break

            x_events_tmp.append(x)
            # Set the coordinate (0,0) at the bottom left corner:
            # NOTE: cAER orgin is at the upper left corner.
            y_events_tmp.append(y)
            ts_events_tmp.append(ts_tot)
            p_events_tmp.append(p)
            
        events = torch.tensor([list(itertools.chain(*x_events_tmp)), 
                               list(itertools.chain(*y_events_tmp)), 
                               list(itertools.chain(*ts_events_tmp)), 
                               list(itertools.chain(*p_events_tmp))], dtype=int)
    return (events)