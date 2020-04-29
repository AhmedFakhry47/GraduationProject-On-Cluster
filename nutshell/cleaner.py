import matplotlib.pyplot as plt
import sys

def read_file(file_dir):
	with open(file_dir,'r') as file:
		lines = file.readlines()
		for line in lines:
			if (line.lower().find('loss') !=-1):	
				loss = float(line.split(':')[-1].split('\n')[0])
				lr   = 1e-3
				yield loss,lr


			elif(line.lower().find('mean') !=-1):
				mAP  = float(line.split('=')[1])
				yield mAP,None

			else:yield None,None


def cleaner(file_dir,where_to_sve):
	file_opened = read_file(file_dir)

	losses = []
	lrates = []
	mAPs   = []
	for i,lr in file_opened:
		if (lr == None):
			if(i==None):continue
			else: mAPs.append(i)
		else:
			losses.append(i)
			lrates.append(lr)

	assert len(mAPs)%2 == 0 , "Mistake in getting mAPs"

	training_mAPs = mAPs[::2]
	test_mAPs = mAPs[1::2]
	n_epochs = range(0,len(losses))
	
	labels = ['Loss','lr','Training_Accuracy','Testing_Accuracy']
	colors = ['.-b','.-k','.-r','.-g']
	All_in = [losses,lrates,training_mAPs,test_mAPs]

	
	for i in range(4):
		plt.subplot(2,2,i+1)
		plt.plot(n_epochs,All_in[i],colors[i],label=labels[i])
		plt.legend(loc='upper left')
	
	plt.savefig(where_to_sve,quality=90,format='png',dpi=100)
	
if __name__ == "__main__":
	if(len(sys.argv) > 3): 
		assert 'Please only add path to the output file'
	file_dir = sys.argv[1]
	where_to_sve = sys.argv[2]
	cleaner(file_dir,where_to_sve)
	
