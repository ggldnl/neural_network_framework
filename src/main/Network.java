package main;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import main.math.CostFunction;
import main.math.MathUtils;
import main.training.DataSet;
import main.training.Match;

/**
 * The Network object is composed of several Layers. It exposes the method to
 * create, train and execute a simple (non convolutional) DFF neural network.
 */
public class Network implements Serializable {

	private static final long serialVersionUID = -6735634455779850654L;

	// stuff for the network
	private Layer[] layers;
	private CostFunction costFunction;

	// stuff for saving/restoring the network
	public static transient final String ext = "net";

	// stuff for console output
	private transient boolean verbose = true;
	private transient ConsoleLogger logger;

	/*
	 * we keep the reference to the file on which we have saved/from which we have
	 * restored the network in order to print it on the console (if verbose is on).
	 */
	protected transient File file;

	/*
	 * instance of the Gson class (singleton with lazy initialization). We can
	 * expect that the object translation in json is not frequent and above all that
	 * it may not occur.
	 */
	protected transient static Gson gson;

	protected static enum State {
		training, executing, restored, saved, not_restored, not_saved
	};

	protected transient State state = State.executing;

	protected transient int current_epoch = 0;
	protected transient int total_epoch = 0;
	protected transient int current_match = 0;
	protected transient int total_matches = 0;

	private Network(Builder builder) {
		this.layers = builder.layers;
		this.costFunction = builder.costFunction;
		logger = new ConsoleLogger(this);
	}

	/*
	 * Builder pattern to build efficiently the network.
	 * 
	 * Usage example:
	 * 
	 * <pre>
	 * Network network = new Network.Builder(input_to_hidden).addLayers(hidden_to_hidden, hidden_to_output)
	 * 		.setLearningRate(0.05f).setCostFunction(CostFunction.Quadratic).compile();
	 * </pre>
	 */
	public static class Builder {

		private List<Layer> layersList = new ArrayList<>();
		private CostFunction costFunction = CostFunction.HalfQuadratic;
		private float learningRate = 0.05f;
		private Layer[] layers;

		public Builder(Layer input_layer) {
			layersList.add(input_layer);
		}

		public Builder addLayer(Layer layer) {
			layersList.add(layer);
			return this;
		}

		public Builder addLayers(Layer... layers) {
			for (Layer layer : layers)
				layersList.add(layer);
			return this;
		}

		public Builder setLearningRate(float learningRate) {
			this.learningRate = learningRate;
			return this;
		}

		public Builder setCostFunction(CostFunction costFunction) {
			this.costFunction = costFunction;
			return this;
		}

		/*
		 * After this, the structure of the network could not be modified (you can still
		 * modify weights and biases if you have a reference of the Layer objects).
		 */
		public Network compile() {

			layers = layersList.toArray(new Layer[layersList.size()]);

			/*
			 * we must set the learning rate for each layer.
			 */
			layers[0].setLearningRate(learningRate);

			/*
			 * while setting the learning rate we can check for error in the dimensions as
			 * well.
			 */
			for (int i = 1; i < layers.length; i++) {
				if (layers[i].getInputNumber() != layers[i - 1].getNeuronNumber())
					throw new IllegalArgumentException(
							String.format("Error in layer[%d]: n_input_%d[%d] != n_neuron_%d[%d].", i, i,
									layers[i].getInputNumber(), i - 1, layers[i].getNeuronNumber()));
				layers[i].setLearningRate(learningRate);
			}
			return new Network(this);
		}
	}

	// getters

	public Layer[] getLayers() {
		return layers;
	}

	public Layer getInputLayer() {
		return layers[0];
	}

	/*
	 * layers has length > 0 for sure.
	 */
	public Layer getOutputLayer() {
		return layers[layers.length - 1];
	}

	// misc

	public void verbose(boolean verbose) {
		this.verbose = verbose;
	}

	// object

	/*
	 * Internally there are two versions for the feedforward method:
	 * 
	 * - the first version is accessible externally. By invoking it, the network
	 * will learn nothing (the backpropagation phase is bypassed) and it is
	 * therefore used after the training;
	 * 
	 * - the second version is not accessible from the outside. It is invoked during
	 * the execution of the train method by the network itself.
	 */

	/**
	 * Feeds the input vector to the network and returns the corresponding output.
	 * The network will learn nothing through the process.
	 * 
	 * @param	input the input float vector
	 * @return	the output float vector (output layer activation)
	 */
	public float[] feedforward(float[] input) {
		return feedforward(input, null);
	}

	/**
	 * Evaluates an input vector and returns the network output. If the target
	 * vector is specified, the network will gather some learning from the
	 * operation.
	 * 
	 * @param	input the input float vector
	 * @param	target the vector by which to correct the network output
	 * @return	the output float vector (output layer activation)
	 */
	private float[] feedforward(float[] input, float[] target) {
		if (input.length != getInputLayer().getInputNumber())
			throw new IllegalArgumentException(String.format("input.lenght[%s] != input_layer.n_neurons[%s]",
					input.length, getInputLayer().getInputNumber()));

		float[] layer_activation = input;
		for (int i = 0; i < layers.length; i++) // avoid iterator creation of the for-each construct
			layer_activation = layers[i].activate(layer_activation);

		if (target != null)
			backpropagate(input, target);

		return layer_activation;
	}

	/*
	 * I prefer to keep the network as generic as possible, so the target will be
	 * represented by a float vector. This method must be used only after a
	 * feedforward, so that an output is first produced to be checked and the
	 * weights adjusted with these operations. It's private, so no worries for
	 * misuses.
	 */
	private void backpropagate(float[] input, float[] target) {
		// we need a delta for each set of weights and each layer has a set of weights
		float[][] delta_weights = null;
		float[] delta_biases = null;
		float[] activation = null;

		int i = layers.length - 1;
		float[] dCdO = costFunction.getDerivative(layers[i].getOutput(), target);

		do {

			// element-wise product between dCdO and the derivative of layers[i].getOutput()
			delta_biases = MathUtils.dCdI(layers[i].getActivationFunction(), layers[i].getOutput(), dCdO);

			if (i > 0)
				activation = layers[i - 1].getOutput();
			else
				activation = input;

			delta_weights = MathUtils.mul(delta_biases, activation);

			layers[i].addWeightsAndBiases(delta_weights, delta_biases);

			// dCdO(l) = dCdO(l+1) * weights(l+1)
			dCdO = MathUtils.mul(delta_biases, layers[i].getWeights());

			i--;
		} while (i >= 0);
	}

	/**
	 * Trains the network on the specified dataset.
	 * 
	 * @param set       the dataset used for the training
	 * @param epochs    epochs
	 */
	public void train(DataSet set, int epochs) {
		train(set, 1, epochs);
	}

	/**
	 * Trains the network on the specified dataset.
	 * 
	 * @param set       the dataset used for the training
	 * @param batchSize batch size
	 * @param epochs    epochs
	 * @throws IllegalArgumentException if batchSize < 1, if epochs < 1 or if batchSize > the set size
	 */
	public void train(DataSet set, int batchSize, int epochs) {

		if (batchSize < 1)
			throw new IllegalArgumentException("Batch size must be more than or equal to one.");
		if (batchSize > set.size())
			throw new IllegalArgumentException(
					"Batch size must be less than or equal to the number of samples in the training dataset.");
		if (epochs < 1)
			throw new IllegalArgumentException("The number of epochs must be more than or equal to one.");

		state = State.training;
		total_epoch = epochs;
		total_matches = set.size();

		int output_layer_length = layers[layers.length - 1].getNeuronNumber();

		for (int e = 0; e < epochs; e++) {

			current_epoch = e;
			current_match = 0;

			for (Match match : set) {

				// preparing the target array
				float[] target = new float[output_layer_length];
				target[match.getLabel()] = 1.0f;

				// giving the target array to the network for reference
				feedforward(match.getInput(), target);

				if (verbose)
					logger.update();

				if (current_match % batchSize == 0)
					update();

				current_match++;
			}
		}

		state = State.executing;
	}

	/*
	 * Updates the layers (each one will adjust its weights and biases).
	 */
	private void update() {
		for (int i = 0; i < layers.length; i++)
			layers[i].adjustWeightsAndBiases();
	}

	/**
	 * Saves the network as an object, serializing it.
	 * 
	 * @param file output file that will contain the serialized network
	 */
	public void save(File file) {

		this.file = file;

		try {

			if (!getExtension(file.getName()).equals(ext))
				throw new IllegalArgumentException("Invalid file type.");

			ObjectOutputStream obj_out = new ObjectOutputStream(new FileOutputStream(file));
			obj_out.writeObject(this);

			state = State.saved;

			obj_out.close();
		} catch (Exception e) {
			state = State.not_saved;
			e.printStackTrace();
		}

		if (verbose)
			logger.update();
	}

	/**
	 * Restores the network deserializing it. This requires an already existing
	 * object, so we should know the structure of the network, create it and then
	 * restore its state from a file that we know for sure has the same structure.
	 * When this is not possible, the static method can be used.
	 * 
	 * @param file the file containing the serialized network
	 */
	public void restore(File file) {

		this.file = file;

		try {

			if (!getExtension(file.getName()).equals(ext))
				throw new IllegalArgumentException("Invalid file type.");

			ObjectInputStream obj_in = new ObjectInputStream(new FileInputStream(file));
			Network network = (Network) obj_in.readObject();

			this.layers = network.layers;
			this.costFunction = network.costFunction;

			state = State.restored;

			obj_in.close();
		} catch (Exception e) {
			state = State.not_restored;
			e.printStackTrace();
		}

		if (verbose)
			logger.update();
	}

	/**
	 * Restores a network from a file. It is not necessary to know its structure.
	 * 
	 * @param file the file containing the serialized network
	 * @return a Network object equals to the one that was serialized into the input file
	 */
	public static Network restoreNetwork(File file) throws IOException {

		if (!getExtension(file.getName()).equals(ext))
			throw new IllegalArgumentException("Invalid file type.");

		ObjectInputStream obj_in = new ObjectInputStream(new FileInputStream(file));
		Network network = null;

		try {
			network = (Network) obj_in.readObject();
		} catch (ClassNotFoundException c) {
			c.printStackTrace();
		}

		obj_in.close();

		return network;
	}

	/*
	 * Finds and returns the extension of the file, if any. The extension is the
	 * last part of the name, the one that succeeds the dot. If multiple extensions
	 * are present, the last one is returned. If no extension is found (i.e.
	 * "filename.") the method will return an empty string. The dot is not included
	 * in the extension returned. We have defined a custom extension for the
	 * serialized network, that is '.net'.
	 */
	private static String getExtension(String filename) {
		return filename.substring(filename.lastIndexOf('.') + 1);
	}

	/**
	 * Returns the network in a JSON format. The JSON will contain all the 
	 * sensible parameters: for each layer - number of neurons in that layer,
	 * number of neurons in the previous layer, activation function, initializer,
	 * weights and biases.
	 * 
	 * @return the JSON string containing all the details of the network.
	 */
	public String toJson() {
		return getGsonInstance().toJson(this);
	}

	/**
	 * Returns the Network object created from a JSON string.
	 * @param json the JSON string
	 * @return the corresponding Network object
	 */
	public static Network fromJson(String json) {
		return getGsonInstance().fromJson(json, Network.class);
	}

	/*
	 * singleton w/lazy initialization
	 */
	private static Gson getGsonInstance() {
		if (gson == null) {
			gson = new GsonBuilder().setPrettyPrinting().create();
		}

		return gson;
	}
}
