import subprocess

test_start_list = [200, 300, 400, 500, 600, 700, 800]
train_size_list = [800, 500, 200, 150, 100, 80, 60, 40, 20]

for test_start in test_start_list:
	for train_size in train_size_list:
		cmd = "python training.py --train_video_path ../data/video1.h5 --label_path ../data/label/video1 --mode same --train_size {} --epoch 20 --batch 8 --test_start {} --base_output_path video1_{}_{} > video1_{}_{}.log".format(train_size, test_start, train_size, test_start, train_size, test_start)
		child = subprocess.Popen(cmd, shell=True)
		child.wait()
