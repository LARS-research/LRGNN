import  os
import re
files = os.listdir('./')
file_names = []
files.sort()
for file in files:
    if '.txt' in file:
        dataset = file.split('_')[1]
        dataset = dataset.upper()
        tmp_data = open('./'+file).readlines()[-1].rstrip()
        expres = tmp_data.split(' ')[-1]
        if 'exp_res' in expres and 'readout' in file:
            # print(file, expres)
            if dataset == 'proteins':
                print('nohup python -u fine_tune.py --data {}       --gpu 1 --num_blocks 12 --num_cells 1 --cell_mode full    --hyper_epoch 20  --arch_filename {}     --cos_lr --BN  1>ft0814_{}.txt &'.format(
                dataset, expres, file[10:]
                ))            
            else:
                print('nohup python -u fine_tune.py --data {}       --gpu 1 --num_blocks 12 --num_cells 1 --cell_mode full    --hyper_epoch 20  --arch_filename {}     --cos_lr --BN --rml2 --rmdropout  1>ft0814_{}.txt &'.format(
                dataset, expres, file[10:]
                ))
        # tmp_data = re.split(':|,', tmp_data)
        # # print(tmp_data)
        # print('{:.02f}({:.02f})'.format(float(tmp_data[4])*100, float(tmp_data[6])*100))