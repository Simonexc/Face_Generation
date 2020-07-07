# Face Generation
The program uses Deep Convolutional Generative Adversarial Network (DCGAN) to create unique new faces. It was trained on the [Sotitoutsi Faces Dataset](https://sortitoutsi.net/graphics/style/1/cut-out-player-faces).

![](assets/Generated%20Face.png)

## Installing
All required libraries can be found in the requirements.txt

You may install all the necessary libraries using the following command ([conda](https://docs.conda.io/en/latest/) is requried)

```conda create --name <env> --file requirements.txt```

## Running the program
Open train.py, set appropriate parameters (under SET PARAMETERS) and run the code to train the model. Once the model is trained (or you opted to check out my model), run test.py file. You don't need to modify anything there.

## Authors
This program was made by Szymon Trochimiak.

## License
This project is provided under the MIT license.
