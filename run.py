#!/usr/bin/env python3

import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn.modules.loss import BCEWithLogitsLoss, CrossEntropyLoss, KLDivLoss, MSELoss
import torch.nn.functional as F
from torch.optim import lr_scheduler
from tqdm import tqdm
import sys


traindir = "training"
testdir = "validation"

save_model_path = 'tartan.pt'
load_model_path = save_model_path


#transformations
train_transforms = transforms.Compose([transforms.Resize((224,224)),
                                       transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225],
    ),
                                       ])
test_transforms = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                      torchvision.transforms.Normalize(
                                          mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225],
    ),
                                      ])

base_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
])
augment_transform = transforms.Compose([
    transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(degrees=15),
    #transforms.RandomCrop(224, padding=4),  # Random cropping, or Resize:
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
])

# ---------------------------------------------------------------------

class NoGoodAugmentedImageFolder(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, augment_transform=None, num_augments=3):
        # Initialize ImageFolder to handle standard loading
        self.image_folder = datasets.ImageFolder(root=root, transform=transform)
        self.augment_transform = augment_transform  # Transform for augmentation
        self.num_augments = num_augments  # Number of augmented images to generate per original image
        self.class_to_idx = self.image_folder.class_to_idx

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, index):
        # Load image and label from ImageFolder
        image, label = self.image_folder[index]

        # Apply additional augmentations on the image
        augmented_images = []
        if self.augment_transform:
            # Apply random affine and other augmentation transforms
            ## for a single one: image = self.augment_transform(image)
        ##return image, label
            for _ in range(self.num_augments):
                # Apply random augmentation to the image
                augmented_image = self.augment_transform(image)
                augmented_images.append(augmented_image.unsqueeze(0))
        # Stack the augmented images into a single tensor (i.e., shape: [num_augments, C, H, W])
        augmented_images_tensor = torch.cat(augmented_images, dim=0)

        return augmented_images_tensor, label

class AugmentedImageFolder(torch.utils.data.Dataset):
    def __init__(self, root_dir, num_augmentations=30, transform=None):
        """
        Args:
            root_dir (str): Directory with class subdirectories of images
            num_augmentations (int): Number of augmented versions to create per image
            transform (callable): Optional transform for augmentations
            base_transform (callable): Optional transform applied to all images including original
        """
        self.original_dataset = datasets.ImageFolder(root=root_dir, transform=None)
        self.num_augmentations = num_augmentations
        self.class_to_idx = self.original_dataset.class_to_idx
        self.base_transform = base_transform
        
        # Default augmentation transform if none provided
        if transform:
          self.transform = transform
        else:
          self.transform = augment_transform

    @property
    def imgs(self):
      ll = []
      #print('ORIG_IMGS = %s' % self.original_dataset.imgs)
      for idx in range(self.__len__()):
        ll.append(self.original_dataset.imgs[idx // (self.num_augmentations + 1)])
      #print('AUG_IMGS = %s' % ll)
      return ll
        
    def __len__(self):
        #print('len returns %d because orig = %d and num_aug = %d' % (len(self.original_dataset) * (self.num_augmentations + 1), len(self.original_dataset), self.num_augmentations))
        return len(self.original_dataset) * (self.num_augmentations + 1)
    
    def __getitem__(self, idx):
        # Calculate which original image this corresponds to
        original_idx = idx // (self.num_augmentations + 1)
        aug_idx = idx % (self.num_augmentations + 1)
        
        # Get original image and label
        image, label = self.original_dataset[original_idx]
        #print('getitem %d, orig=%d aug=%d = %s' % (idx, original_idx, aug_idx, label))
        
        # If it's the first version, return the original with only base transform
        if aug_idx == 0:
            if self.base_transform:
                image = self.base_transform(image)
            return image, label
        
        # Otherwise, apply augmentation
        if self.transform:
            image = self.transform(image)            
        return image, label





# ---------------------------------------------------------------------
#datasets
#train_data = datasets.ImageFolder(traindir,transform=train_transforms)
train_data = AugmentedImageFolder(traindir,transform=train_transforms)
#test_data = datasets.ImageFolder(testdir,transform=test_transforms)
test_data = AugmentedImageFolder(testdir,transform=test_transforms)

class_names = list(train_data.class_to_idx.keys())
print(class_names)
# don't assume they are in order, so find each index in turn:
num_classes = len(train_data.class_to_idx)
for idx in range(num_classes):
  print('classes = %s' % [x for x in train_data.class_to_idx if train_data.class_to_idx[x]==idx][0])

#dataloader
trainloader = torch.utils.data.DataLoader(train_data, shuffle = True, batch_size=16) # also num_workers=4
testloader = torch.utils.data.DataLoader(test_data, shuffle = True, batch_size=16)


def make_train_step(model, optimizer, loss_fn):
  def train_step(x,y):
    # make prediction
    yhat = model(x)

    # enter train mode
    model.train()

    # compute loss
    loss = loss_fn(yhat,y)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    #optimizer.cleargrads()

    return loss
  return train_step



device = "cuda" if torch.cuda.is_available() else "cpu"
print('Running on %s' % device)

model = models.resnet18(weights='DEFAULT') # same as weights='IMAGENET1K_V1' or ResNet18_Weights.DEFAULT, was (pretrained=True)

#freeze all params
for params in model.parameters():
  params.requires_grad_ = False

#add a new final layer
nr_filters = model.fc.in_features  #number of input features of last layer
model.fc = nn.Linear(nr_filters, num_classes)

if load_model_path:
  print('Loading model weights from %s' % load_model_path)
  path_loader = torch.load(load_model_path)
  model.load_state_dict(path_loader)
  model.eval()

model = model.to(device)




#loss
#loss_fn = BCEWithLogitsLoss() #binary cross entropy with sigmoid, so no need to use sigmoid in the model
loss_fn = CrossEntropyLoss()
#loss_fn = KLDivLoss()
#loss_fn = MSELoss()

#optimizer
optimizer = torch.optim.Adam(model.fc.parameters())

#train step
train_step = make_train_step(model, optimizer, loss_fn)



def train():


  losses = []
  val_losses = []

  epoch_train_losses = []
  epoch_test_losses = []

  n_epochs = 10
  early_stopping_tolerance = 3
  early_stopping_threshold = 0.03

  for epoch in range(n_epochs):
    epoch_loss = 0
    for i ,data in tqdm(enumerate(trainloader), total = len(trainloader)): #iterate over batches
      x_batch , y_batch = data
      x_batch = x_batch.to(device) #move to gpu
      #y_batch = y_batch.unsqueeze(1).float() #convert target to same nn output shape
      y_batch = y_batch.to(device) #move to gpu


      loss = train_step(x_batch, y_batch)
      epoch_loss += loss/len(trainloader)
      losses.append(loss)

    epoch_train_losses.append(epoch_loss)
    print('\nEpoch : {}, train loss : {}'.format(epoch+1,epoch_loss))

    #validation doesnt requires gradient
    with torch.no_grad():
      cum_loss = 0
      for x_batch, y_batch in testloader:
        x_batch = x_batch.to(device)
        #y_batch = y_batch.unsqueeze(1).float() #convert target to same nn output shape
        y_batch = y_batch.to(device)

        #model to eval mode
        model.eval()

        yhat = model(x_batch)
        val_loss = loss_fn(yhat,y_batch)
        cum_loss += loss/len(testloader)
        val_losses.append(val_loss.item())


      epoch_test_losses.append(cum_loss)
      print('Epoch : {}, val loss : {}'.format(epoch+1,cum_loss))

      best_loss = min(epoch_test_losses)

      #save best model
      if cum_loss <= best_loss:
        best_model_wts = model.state_dict()
      #print('EARLY BREAK')
      #break # XXX

      #early stopping
      early_stopping_counter = 0
      if cum_loss > best_loss:
        early_stopping_counter +=1

      if (early_stopping_counter == early_stopping_tolerance) or (best_loss <= early_stopping_threshold):
        print("\nTerminating: early stopping")
        break #terminate training

  #load best model
  model.load_state_dict(best_model_wts)

  if save_model_path:
    print('Saving model weights to %s' % save_model_path)
    torch.save(model.state_dict(), save_model_path)



def inference(test_data):
  idx = torch.randint(1, len(test_data), (1,))
  sample = torch.unsqueeze(test_data[idx][0], dim=0).to(device)
  actual_class = test_data[idx][1]

  # binary classification uses sigmod, multiple classification uses softmax
  #if torch.sigmoid(model(sample)) < 0.5:
  #  print("Prediction : Cat")
  #else:
  #  print("Prediction : Dog")

  # Assuming `model` is your pre-trained model and `input_image` is the image to classify
  logits = model(sample)  # Output of the model (raw logits)
  #print('logits = %s' % logits)
  #for nn in range(num_classes):
    #print('logits %d = %s' % (nn, logits[nn]))
    ##print('logits %d item = %s' % (nn, logits[nn].item()))

  # Apply softmax to get probabilities
  probabilities = F.softmax(logits, dim=1)
  torch.set_printoptions(profile="full")
  #print('probabilities = %s' % probabilities)
  #print('probabilities = %s' % probabilities.item())
  #for nn in range(5):
    #print(probabilities[nn])
    ##print(probabilities[nn].item())

  # Get the predicted class (index of the class with the highest probability)
  # max() gets max value, argmax() gets index of max probability
  predicted_class = torch.argmax(probabilities, dim=1).item()
  predicted_label = class_names[predicted_class]  # Convert the tensor to a Python number
  print('test on idx %d = class %d file %s' % (idx, actual_class, test_data.imgs[idx][0]))
  print('predicted class = %s' % predicted_class)
  print('predicted label = %s' % predicted_label)
  if predicted_class == actual_class:
    return True
  return False


if not load_model_path:
  train()

correct=0
for nnn in range(30):
  if inference(test_data):
    correct += 1
print('accuracy = %f%%' % (100.0*correct/30))
#print(test_data.imgs)
