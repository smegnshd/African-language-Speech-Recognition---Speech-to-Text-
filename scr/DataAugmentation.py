import muda
import jams
import glob
import pathlib
class AudioManipulation:
    """This class implementes the Data Augmentation part of the Paper
    Deep Convolutional Neural Networks and DAta Augmentation for Environmental
    Sound Classification
    
    It implements Time Stretching(TS), Pitch Shifting twice(PS1, PS2) Dyanmic Range Compression(DRC), 
    Background Nose(BG). 
    The implementation utilizes MUDA library, and as a comparsion there could be a part that does jams checking 
    https://github.com/justinsalamon/UrbanSound8K-JAMS
    https://jams.readthedocs.io/en/stable/ and https://jams.readthedocs.io/en/stable/examples.html
    https://muda.readthedocs.io/en/stable/index.html
    """

    def __init__(self):
        pass

    def time_stretching(self, file, suffix_file, stretching_array):
        jam = jams.JAMS()
        jam = muda.load_jam_audio(jam, file)
        pth = pathlib.Path(file)
        parent = pth.parent
        file_name = pth.stem
        suffix = pth.suffix
        bg_transform = muda.deformers.TimeStretch(stretching_array)
        for i, jam_out in enumerate(bg_transform.transform(jam)):
             muda.save('{}/{}_{}_{:02d}{}'.format(parent, file_name, suffix_file,i, suffix),'{}/{}_{}_{:02d}.{}'.format(parent, file_name, suffix_file,i, "jams"),jam_out)

    def pitch_shifting(self, file, suffix_file, shifting_array):
        jam = jams.JAMS()
        jam = muda.load_jam_audio(jam, file)
        pth = pathlib.Path(file)
        parent = pth.parent
        file_name = pth.stem
        suffix = pth.suffix
        bg_transform = muda.deformers.PitchShift(shifting_array)
        for i, jam_out in enumerate(bg_transform.transform(jam)):
             muda.save('{}/{}_{}_{:02d}{}'.format(parent, file_name, suffix_file,i, suffix),'{}/{}_{}_{:02d}.{}'.format(parent, file_name, suffix_file,i, "jams"),jam_out)

    def dynamic_range_compression(self, file, suffix_file, presets):
        jam = jams.JAMS()
        jam = muda.load_jam_audio(jam, file)
        pth = pathlib.Path(file)
        parent = pth.parent
        file_name = pth.stem
        suffix = pth.suffix
        bg_transform = muda.deformers.DynamicRangeCompression(presets)
        for i, jam_out in enumerate(bg_transform.transform(jam)):
             muda.save('{}/{}_{}_{:02d}{}'.format(parent, file_name, suffix_file,i, suffix),'{}/{}_{}_{:02d}.{}'.format(parent, file_name, suffix_file,i, "jams"),jam_out)
    def background_noise_addition(self, file, suffix_file, bg_noises):
        # create an empty jam
        jam = jams.JAMS()
        jam = muda.load_jam_audio(jam, file)
        pth = pathlib.Path(file)
        parent = pth.parent
        file_name = pth.stem
        suffix = pth.suffix
        bg_transform = muda.deformers.BackgroundNoise(n_samples=1, files=bg_noises)
        for i, jam_out in enumerate(bg_transform.transform(jam)):
             muda.save('{}/{}{}{:02d}{}'.format(parent, file_name, suffix_file,i, suffix),'{}/{}_{}_{:02d}.{}'.format(parent, file_name, suffix_file,i, "jams"),jam_out)

if __name__ == "__main__":
    y = AudioManipulation()
    # We would like take fold1_20 and so on and do that iteratively. 
    # And then again. and again. and again. 
    # this was a remnant of something.
    # folders = ["fold10_20/", "fold1_20/", "fold2_20/", "fold3_20/", "fold4_20/", "fold5_20/", "fold6_20/", "fold7_20/", "fold8_20/", "fold9_20/"]
    files = glob.glob("AmharicSTT/*/*.wav")
    bg_noises=["173955__saphe__street-scene-3.wav","207208__jormarp__high-street-of-gandia-valencia-spain.wav","268903__yonts__city-park-tel-aviv-israel.wav", "background_noise_150993__saphe__street-scene-1.wav"]
    for i in files:
        # now this is all the files.
        y.time_stretching(i, "ts", [0.81, 0.93, 1.07, 1.23])
        y.pitch_shifting(i, "ps", [-3.5, -2.5, -2, -1, 1, 2, 2.5, 3.5])
        y.dynamic_range_compression(i, "drc", ["radio", "film standard", "speech", "music standard"])
        y.background_noise_addition(i, "bn", bg_noises)