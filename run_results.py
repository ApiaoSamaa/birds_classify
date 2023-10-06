import torch
import contextlib
import warnings
from models.builder import MODEL_GETTER
from utils.costom_logger import timeLogger
from utils.config_utils import load_yaml, build_record_folder, get_args
warnings.simplefilter("ignore")
import os
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import torch

class TestImageDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 istrain: bool,
                 root: str,
                 data_size: int,
                 return_index: bool = False):
        """ basic information """
        self.root = root
        self.data_size = data_size
        self.return_index = return_index

        """ declare data augmentation """
        normalize = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
        if istrain:
            self.transforms = transforms.Compose([
                        transforms.Resize((510, 510), Image.BILINEAR),
                        transforms.RandomCrop((data_size, data_size)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=0.1),
                        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
                        transforms.ToTensor(),
                        normalize
                ])
        else:
            self.transforms = transforms.Compose([
                        transforms.Resize((510, 510), Image.BILINEAR),
                        transforms.CenterCrop((data_size, data_size)),
                        transforms.ToTensor(),
                        normalize
                ])

        """ read all data information """
        self.data_infos = self.getDataInfo(root)

    # TODO
    def getDataInfo(self, root):
        data_infos = []
        files = [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]
        number_of_files = len(files)
        print("[test pictures] number:", number_of_files)
        for _, file in enumerate(files):
            data_path = root+"/"+file
            data_infos.append({"path":data_path, "filename":file}) # TODO
        return data_infos

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index):
        # get data information.
        image_path = self.data_infos[index]["path"]
        filename = self.data_infos[index]["filename"]
        # read image by opencv.
        img = cv2.imread(image_path)
        img = img[:, :, ::-1] # BGR to RGB.
        
        # to PIL.Image
        img = Image.fromarray(img)
        img = self.transforms(img)
        
        if self.return_index:
            # return index, img, sub_imgs, label, sub_boundarys
            return index, img, filename
        
        return img, filename

def build_loader(args):
    _ , train_loader = None, None
    val_set, val_loader = None, None
    if args.val_root is not None:
        val_set = TestImageDataset(istrain=False, root=args.val_root, data_size=args.data_size, return_index=True)
        val_loader = torch.utils.data.DataLoader(val_set, num_workers=1, shuffle=True, batch_size=args.batch_size)

    return train_loader, val_loader

def set_environment(args, tlogger):
    
    print("Setting Environment...")

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    ### = = = =  Dataset and Data Loader = = = =  
    tlogger.print("Building Dataloader....")
    
    _ , val_loader = build_loader(args)
    
    if val_loader is not None:
        print("    Validation Samples: {} (batch: {})".format(len(val_loader.dataset), len(val_loader)))
    else:
        print("    Validation Samples: 0 ~~~~~> [Only Training]")
    tlogger.print()

    ### = = = =  Model = = = =  
    tlogger.print("Building Model....")
    model = MODEL_GETTER[args.model_name](
        use_fpn = args.use_fpn,
        fpn_size = args.fpn_size,
        use_selection = args.use_selection,
        num_classes = args.num_classes,
        num_selects = args.num_selects,
        use_combiner = args.use_combiner,
    ) # about return_nodes, we use our default setting
    if args.pretrained is not None:
        checkpoint = torch.load(args.pretrained, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0
    model.to(args.device)
    tlogger.print()
 
    ### = = = =  Optimizer = = = =  
    tlogger.print("Building Optimizer....")
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.max_lr, nesterov=True, momentum=0.9, weight_decay=args.wdecay)
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.max_lr)

    if args.pretrained is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    tlogger.print()


    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
        # TODO what does this mean?
        amp_context = torch.cuda.amp.autocast
    else:
        scaler = None
        amp_context = contextlib.nullcontext

    return None , val_loader, model, optimizer, None , scaler, amp_context, start_epoch



def convert_gif_to_jpg(directory_path):
    directory = Path(directory_path)
    # Iterate through all .jpg files
    for file_path in directory.glob("*.jpg"):
        with Image.open(file_path) as img:
            if img.format == "GIF":
                # Convert GIF to JPEG
                converted_img = img.convert('RGB')
                converted_img.save(file_path, "JPEG")

@torch.no_grad()
def predict_and_save(args, model, val_loader):
    print("Start Printing Birds Prediction ...")
    model.eval()
    results = []    
    with torch.no_grad():
        for _ , (ids, data, filename) in enumerate(val_loader):
            data = data.to(args.device)
            outs = model(data)
            this_name = "combiner"
            pre = outs["comb_outs"].argmax(dim=-1).cpu().numpy()
            print(list(zip(filename, pre)))
            results.extend(list(zip(filename, pre)))
    # msg += "Project: {}, Experiment: {}\n".format(args.project_name, args.exp_name)
    # msg += "Samples: {}\n".format(len(val_loader.dataset))
    msg += "\n"
    for result in results:
        msg += "{} {}\n".format(result[0], result[1])
    with open(args.save_dir + "eval_results.txt", "w") as ftxt:
        ftxt.write("58120309 王玟雯")
    with open(args.save_dir + "eval_results.txt", "a") as ftxt:
        ftxt.write(msg)
  
def main(args, tlogger):
    _ , val_loader, model, _, _, _, _, _ = set_environment(args, tlogger)
    predict_and_save(args, model, val_loader)

if __name__ == "__main__":

    tlogger = timeLogger()
    tlogger.print("Reading Config...")
    args = get_args()
    assert args.c != "", "Please provide config file (.yaml)"
    load_yaml(args, args.c)
    build_record_folder(args)
    tlogger.print()
    convert_gif_to_jpg(args.val_root)
    main(args, tlogger)