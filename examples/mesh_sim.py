import numpy as np
import pygsound as ps
from wavefile import WaveWriter, Format
from scipy import signal, io
import librosa
import wavfile
from playsound import playsound
import matplotlib
import matplotlib.pyplot as plt


def main():
    # Simulation using .obj file (and an optional .mtl file)
    ctx = ps.Context()
    ctx.diffuse_count = 20000
    ctx.specular_count = 2000
    ctx.channel_type = ps.ChannelLayoutType.stereo
    obj_path = "./00ad8345-45e0-45b3-867d-4a3c88c2517a/house.obj"
    mesh1 = ps.loadobj(obj_path)
    scene = ps.Scene()
    scene.setMesh(mesh1)

    src_coord = [1, 1, 1]
    lis_coord_1 = [-2, 6, 1]
    lis_coord_2 = [1, 1.2, 1]

    src = ps.Source(src_coord)
    src.radius = 0.01
    src.power = 1.0

    lis = ps.Listener(lis_coord_1)
    lis.radius = 0.01

    res = scene.computeIR([src], [lis], ctx)  # you may pass lists of sources and listeners to get N_src x N_lis IRs
    audio_data1 = np.array(res['samples'][0][0])  # the IRs are indexed by [i_src, i_lis, i_channel]

    print(np.array(res['samples'][0][0]))
    with WaveWriter('test1.wav', channels=audio_data1.shape[0], samplerate=int(res['rate'])) as w1:
        w1.write(audio_data1)
        print("IR using .obj input written to test1.wav.")

    # Simulation using the same room with a different listener position:
    lis2 = ps.Listener(lis_coord_2)
    res2 = scene.computeIR([src], [lis2], ctx)  # you may pass lists of sources and listeners to get N_src x N_lis IRs
    audio_data2 = np.array(res2['samples'][0][0])  # the IRs are indexed by [i_src, i_lis, i_channel]

    with WaveWriter('test2.wav', channels=audio_data2.shape[0], samplerate=int(res2['rate'])) as w2:
        w2.write(audio_data2)
        print("IR using .obj input written to test2.wav.")

    sample_res = wavfile.read("audio.wav", normalized=True, forcestereo=True)
    impulse_res = wavfile.read("test2.wav", normalized=True, forcestereo=True)
    sr = sample_res[1]
    ir = impulse_res[1]
    out_0 = signal.fftconvolve(sr[:, 0], ir[:, 0])
    out_1 = signal.fftconvolve(sr[:, 1], ir[:, 1])
    out = np.vstack((out_0, out_1)).T * 32000
    # out = (out - np.min(sr)) / (np.max(sr) - np.min(sr))
    print(sample_res[1].max() / out[1].max())

    wavfile.write("test-lobby.wav", sample_res[0], out, normalized=True)


if __name__ == '__main__':
    main()
