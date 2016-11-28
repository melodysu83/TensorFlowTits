from controlmenu import Menu

def main():	
	# choose machine learning model	
	model =  2  # (0: MLP,      1: RNN,     2: CNN)
	mode =   2  # (0: training, 1: testing, 2: real world implementation)  

	# setup
	menu = Menu(mode)	
	
	# start running
	if model == menu.MLP: 			
		menu.MLP_Process(15,100)     # epochs, batchsize

	elif model == menu.RNN: 
		menu.RNN_Process(15,128,28)   # epochs, batchsize, chunk
	
	elif model == menu.CNN:
		menu.CNN_Process(15,128,0.8) # epochs, batchsize, keep_rate
	
	
	
if __name__ == "__main__":
    main()
