from controlmenu import Menu

def main():	
	# choose machine learning model
	# (0: MLP, 1: RNN, 2: CNN)
	model =  2      
	training = True
	testing = True	

	# setup
	menu = Menu(training,testing)	

	# start running
	if model == menu.MLP: 			
		menu.MLP_Process(15,100)    # epochs, batchsize

	elif model == menu.RNN: 
		menu.RNN_Process()
	
	elif model == menu.CNN:
		menu.CNN_Process(15,128,0.8) # epochs, batchsize, keep_rate
	
	
if __name__ == "__main__":
    main()
