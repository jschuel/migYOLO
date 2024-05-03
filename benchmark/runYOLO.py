'''
Script is a part of runBenchmark.sh and is designed to process and perform Migdal skims alongside of a screen session running downsample.py. 
Use runBenchmark.sh to run this
'''
import os
os.sys.path.append('../pipeline')
from pipeline import yolo
from ultralytics import YOLO
import time
import subprocess
from readYAML import read_config_file

'''Check if screen session is running'''
def is_screen_running(screen_name):
    try:
        # Run the 'screen -ls' command and capture the output
        output = subprocess.check_output(["screen", "-ls"]).decode("utf-8")
        # Check if the screen session is in the output
        return screen_name in output
    except subprocess.CalledProcessError:
        return False

'''Save timestamps each time a file s processed to the "timestamp/" directory'''
def save_timestamp():
    fname = 'timestamps/%s.txt'%(time.time())
    with open(fname, "w") as f:
        f.write(str(time.time()))

def main():
    '''Read configuration from yoloConf.yaml'''
    conf = read_config_file('yoloConf.yaml')['yoloConf']
    model = YOLO(conf['model'])
    '''save initial timestamp'''
    save_timestamp()

    '''Wait 3 seconds to ensure screen session runs'''
    time.sleep(3)

    '''Define fis so we can initialize while loop condition'''
    fis = [conf['downsample_path']+'/'+fi for fi in sorted(os.listdir(conf['downsample_path'])) if '.npy' in fi]

    '''While the downsample script is running, or files remain in fis, use YOLO to process downsampled images.
    After YOLO completes it deletes the downsampled image as well as the pngs input into YOLO'''
    while is_screen_running('Downsample') or len(fis) > 0:
        fis = [conf['downsample_path']+'/'+fi for fi in sorted(os.listdir(conf['downsample_path'])) if '.npy' in fi]    
        for fi in fis:
            t0 = time.time()
            try: #Run YOLO
                yolo(infile=fi, outpathbase = conf['outpath'], model=model, create_pngs = False, vignetting_correction_file=conf['vignetting_map'], calibration_file=conf['calibration_file'], png_threshold=conf['png_threshold'], png_saturation=conf['png_saturation'], migdal_cut = conf['migdal_cut'], delete_output = conf['delete_yolo_output'])
            except: #To prevent instances where the png images haven't been generated before calling YOLO
                print("Wait 1.5 second")
                time.sleep(1.5)
                yolo(infile=fi, outpathbase = conf['outpath'], model=model, create_pngs = False, vignetting_correction_file=conf['vignetting_map'], calibration_file=conf['calibration_file'], png_threshold=conf['png_threshold'], png_saturation=conf['png_saturation'], migdal_cut = conf['migdal_cut'], delete_output = conf['delete_yolo_output'])
        
            tf = time.time()
            print(tf-t0)
            save_timestamp()
        fis = [conf['downsample_path']+'/'+fi for fi in sorted(os.listdir(conf['downsample_path'])) if '.npy' in fi] #check again as path needs to be cleared for script to terminate
if __name__ == '__main__':
    main()
