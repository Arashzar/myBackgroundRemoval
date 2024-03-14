
import os
import tqdm

# from libs.strings import *
from libs.networks import model_detect
import libs.preprocessing as preprocessing
import libs.postprocessing as postprocessing



def __work_mode__(path: str):

    if os.path.isfile(path):  # Input is file
        return "file"
    if os.path.isdir(path):  # Input is dir
        return "dir"
    else:
        return "no"


def __save_image_file__(img, file_name, output_path, wmode):

    # create output directory if it doesn't exist
    folder = os.path.dirname(output_path)
    if folder != '':
        os.makedirs(folder, exist_ok=True)
    if wmode == "file":
        file_name_out = os.path.basename(output_path)
        if file_name_out == '':
            # Change file extension to png
            file_name = os.path.splitext(file_name)[0] + '.png'
            # Save image
            img.save(os.path.join(output_path, file_name))
        else:
            try:
                # Save image
                img.save(output_path)
            except OSError as e:
                if str(e) == "cannot write mode RGBA as JPEG":
                    raise OSError("Error!!!  Please indicate the correct extension file, for ex: .png")
                else:
                    raise e
    else:
        # Change file extension to png
        file_name = os.path.splitext(file_name)[0] + '.png'
        # Save image
        img.save(os.path.join(output_path, file_name))

def process(input_path, output_path, model_name="u2netp",
            preprocessing_method_name="bbd-maskrcnn", postprocessing_method_name="rtb-bnb2"):

    if input_path is None or output_path is None:
        raise Exception("Bad parameters! Please specify input path and output path.")

    model = model_detect(model_name)  # Load model
    if not model:

        model_name = "u2net"  # If the model line is wrong, select the model with better quality.
        model = model_detect(model_name)  # Load model
    preprocessing_method = preprocessing.method_detect(preprocessing_method_name)
    postprocessing_method = postprocessing.method_detect(postprocessing_method_name)
    wmode = __work_mode__(input_path)  # Get work mode
    if wmode == "file":  # File work mode
        image = model.process_image(input_path, preprocessing_method, postprocessing_method)
        __save_image_file__(image, os.path.basename(input_path), output_path, wmode)
    elif wmode == "dir":  # Dir work mode
        # Start process
        files = os.listdir(input_path)
        for file in tqdm.tqdm(files, ascii=True, desc='Remove Background', unit='image'):
            file_path = os.path.join(input_path, file)
            image = model.process_image(file_path, preprocessing_method, postprocessing_method)
            __save_image_file__(image, file, output_path, wmode)
    else:
        raise Exception("Bad input parameter! Please indicate the correct path to the file or folder.")


def cli():
    """CLI"""



    # input_path = args.input_path
    input_path = r"./docs/imgs/input/"
    # output_path = args.output_path
    output_path = r'./docs/imgs/output/'
    # model_name = args.model_name
    model_name = 'u2netp'
    # preprocessing_method_name = args.preprocessing_method_name
    preprocessing_method_name = 'mobile-net-model'
    # postprocessing_method_name = args.postprocessing_method_name
    postprocessing_method_name ='rtb-bnb'

    if model_name == "test":
        print(input_path, output_path, model_name, preprocessing_method_name, postprocessing_method_name)
        # processTestModel(input_path, output_path, model_name, preprocessing_method_name, postprocessing_method_name)
    else:
        print(input_path,output_path,model_name,preprocessing_method_name,postprocessing_method_name)
        process(input_path, output_path, model_name, preprocessing_method_name, postprocessing_method_name)


if __name__ == "__main__":
    cli()
