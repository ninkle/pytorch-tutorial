import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader

import os 
import time
from tqdm import tqdm  # this is a nice library for making nice looking progress bars

class Trainer(object):
	def __init__(self, model, save_dir):
		self.model = model
		self.save_dir = save_dir

	def train(self, train, val, epochs, batch_size, log_per_batches, learning_rate, device):

		run_id = time.clock()  # give this training run a unique id, we'll use this later when we save our model
		self.model.to(device)

		# batch data and pass to iterator
		trainloader = DataLoader(train, batch_size=batch_size, shuffle=True)  # pass training data to pytorch's data loader (an iterator)

		if val is not None:
			valloader = DataLoader(val, batch_size=batch_size, shuffle=True)  # pass validation data to pytorch's dataloader (an iterator)

		# stuff for you to play with!
		loss_fn = nn.CrossEntropyLoss()  # you will need to pick an appropriate loss function for the model you're building
		optimizer = optim.Adam(self.model.parameters(), learning_rate)  # adam is typically the standard choice but feel free to play around w this

		# keep a running loss that we'll average over - this gives us a smoother estimate on the loss
		running_train_loss = 0
		running_val_loss = 0

		# we'll also keep track of our previous loss on the validation set and use this figure out when to stop training 
		prev_val_loss = 0

		# begin train loop
		for epoch in range(epochs):
			print("Epoch %s" % epoch)

			train_bar = tqdm(enumerate(trainloader, 1), total=len(trainloader))  # wrap with tqdm to make a pretty progress bar

			# iterate over training data
			for i, data in train_bar:
				x, targets = data

				optimizer.zero_grad()  # call to zero gradients

				predictions = self.model(x)  # pass x to model and return predictions
				loss = loss_fn(predictions, targets)  # calculate loss on this batch of predictions
				running_train_loss += loss  # add to our running total

				loss.backward()  # backpropogate the loss and update model parameters
				optimizer.step()  # update optimizer paraeters

				if i % log_per_batches == 0:  # print every `log_per_batches` batches
					avg_train_loss = running_train_loss / log_per_batches  # average the loss 
					train_bar.set_description("Training Loss: %.3f" % avg_train_loss)  # update the description of the tqdm progress bar 
					running_train_loss = 0  # reset running loss to zero

			# iterate over validation data
			if val is not None:

				val_bar = tqdm(enumerate(valloader, 1), total=len(valloader))  # wrap with tqdm to make a pretty progress bar
				
				with torch.no_grad():  # use `torch.no_grad()` so information about our val set does not get backpropgated
					for i, data in val_bar:
						x, targets = data

						# everything below is the same as the training loop, but notice that we don't need to backprop loss/update optimizer
						predictions = self.model(x)
						loss = loss_fn(predictions, targets)
						running_val_loss += loss

					# average over the entire validation sett
					avg_val_loss = running_val_loss / len(valloader)
					val_bar.set_description("Validation Loss: %.3f" % avg_val_loss)

				# set an early stopping condition
				# there's a variety of other conditions we can use, but this is nice and simple
				if avg_val_loss > prev_val_loss:  # model is still improving!
					state = {"state_dict": self.model.state_dict()}
					if self.save_dir is not None:
						try:
							os.makedirs(save_dir)  # make save_dir if save_dir does not already exist
						except FileExistsError:  # save_dir already exists
							pass

						# save path will be `save_dir/run_id/epoch_{epoch_number}`
						save_path = os.path.join(self.save_dir(os.path.join(run_id, "epoch_"+str(epoch))))  
						torch.save(state, save_path) # save model

				else:  # model is beginning to overfit
					return  # so we stop training








					







		
