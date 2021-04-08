# Face recognition- Are you wearing a mask or not?
<h3>A Machine Learning algorithm capable of identifying people who are using a mask or similar.</h3>
<h6>This project aims to apply the knowledge acquired in the artificial intelligence and computational nanodregree by <a href="https://www.fiap.com.br/" target="_blank">FIAP</a>.</h6>

<p>
The script presented is a supervised machine learning, where a database (in this case images) is presented, providing a reference of what is "right" and "wrong", creating a model based on this data.
</p>

# Requirements
<ul>
  <li><a href="https://www.python.org/downloads/release/python-389/">Python 3.8</a> -> Developed with version 3.8, but has not been tested in other versions.</li>
  <li><a href="https://www.tensorflow.org/">Tensorflow</a>
    <blockquote>
      <code>pip install --upgrade tensorflow</code>
    </blockquote>
  </li>
  <li><a href="https://keras.io/">Keras</a>
    <blockquote>
      <code>pip install keras</code>
    </blockquote>
  </li>
  <li><a href="https://numpy.org/">Numpy</a>
    <blockquote>
      <code>pip install numpy scipy</code>
    </blockquote>
  </li>
  <li><a href="https://scikit-learn.org/">Scikit Learn</a>
    <blockquote>
      <code>pip install scikit-learn</code>
    </blockquote>
  </li>
  <li><a href="https://pypi.org/project/imutils/">Imutils</a>
    <blockquote>
      <code>pip install imutils</code>
    </blockquote>
  </li>
  <li><a href="https://pillow.readthedocs.io/">Pillow</a>
    <blockquote>
      <code>pip install pillow</code>
    </blockquote>
  </li>
  <li><a href="https://www.h5py.org/">h5py</a>
    <blockquote>
      <code>pip install h5py</code>
    </blockquote>
  </li>
  <li><a href="https://matplotlib.org/">Matplotlib</a>
    <blockquote>
      <code>pip install matplotlib</code>
    </blockquote>
  </li>
</ul>
For development I used Anancoda and Jupyter Notebook, but use the development tools you prefer.

# Dataset

# Training
<p>
You can train your model, just build your database and do the training on it.
</p>
<p>
<h3>Training variables</h3>
<p><b>learning_rate:</b> Initial learning rate.</p>
  <p>
    <blockquote>
      The learning rate adjusts the optimization algorithm that determines the size of the step in each iteration, adapting it so that it has the minimum of loss. The standard learning rate is 0.001, but you can do a variable learning rate throughout the learning process, <a href="https://keras.io/api/optimizers/learning_rate_schedules/">read more</a>.
    </blockquote>
  </p>
<p><b>training_size:</b> Number of times the model will train with the dataset.
  <p>
    <blockquote>
    The amount of training can vary according to your dataset. If your dataset is very large, you <b>maybe</b> able to get good results by doing the training a few times. Start with a value of around 2 ~ 5% of your dataset (for 1000 samples, train 20 ~ 50 times), and check the "Training cycle (Epoch) X Loss / Accuracy" graph to see if the model has met its objectives.
    </blockquote>
  </p>
</p>
<p><b>batch_size:</b> The batch size defines the number of samples that will be propagated over the network. The algorithm takes the first N samples (from the 1st to the Nª) of the training dataset and trains the network. Then, he takes the second N samples (from (N+1)ª to (2*N)ª) and trains the network again. N is the batch_size.
  <p>
    <blockquote>
    It is recommended to use a batch size minor that number of all samples because requires less memory. Since you train the network using fewer samples, the general training procedure requires less memory. Usually the networks train faster with mini-lots.
    </blockquote>
  </p>
</p>
<p><b>images_path:</b> Path with positive and negative images for training.</p>
</p>

# Image Application

<p>
  To test the trained model using an image, run script <a href"https://github.com/hconcessa/facemask-recognition/blob/main/src/application/detect-by-image.py">detect-by-image.py</a>.
To use your image, change the directory indicated in the imageExamplePath variable of the script to the image you are going to use.
</p>

<h4>Before running the script:</h4>
<img src="https://github.com/hconcessa/facemask-recognition/blob/main/examples/example1.png" width="50%"></img>
<h4>After running the script:</h4>
<img src="https://github.com/hconcessa/facemask-recognition/blob/main/examples/Results/Example1-after-run-script.png" width="50%"></img>

# Camera Application
