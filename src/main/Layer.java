package main;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.Serializable;

import main.math.ActivationFunction;
import main.math.Initializer;
import main.math.MathUtils;

/**
 * This class represents a single layer in the network. Each Layer must manage
 * the weights and biases from which it is conceptually formed, represented as a
 * matrix and vector respectively. The weights matrix will have dimension <i>(n
 * neurons × n input)</i>, the biases vector will have length <i>(n
 * neurons)</i>. Both of them will be initialized by the Initializer immediately
 * after the layer creation.
 * 
 * Each Layer maintains two auxiliary data structures, <i>delta_weights</i> and
 * <i>delta_biases</i>. Each element of the matrix delta_weights represents the
 * change to be made on the respective element of the weights matrix; the same
 * for the delta_biases vector.
 * 
 * The update is defined from outside the layer, at a network level, so it might
 * not always happen after a feedforward as it depends on the network
 * hyperparameters. The value of the deltas is accumulated in the auxiliary data
 * structures at each feedforward of the training phase by invoking the
 * <i>addWeightsAndBiases</i> method. When the method
 * <i>adjustWeightsAndBiases</i> is invoked from outside, changes are finalized.
 * 
 */
public class Layer implements Serializable {

	private static final long serialVersionUID = -6284308204896056741L;

	// object
	private int n_neurons;
	private int n_input;
	private float learningRate;

	private ActivationFunction activationFunction;
	private Initializer initializer;

	private transient float[] activations;
	private float[][] weights;
	private float[] biases;

	// weights and biases update
	private transient float[][] delta_weights;
	private transient float[] delta_biases;
	private int batch;

	/**
	 * Used to build a layer for an untrained network. The number of neurons of the
	 * previous layer and that of the ones in the current layer are required. As
	 * there are no weights and biases to use (we are not restoring a network
	 * previously saved), they are initialized here by the Initializer. They can be
	 * initialized to zero or generated following various approaches, depending on
	 * the chosen Initializer.
	 * 
	 * Both the activation function and the initializer can be set through the
	 * constructor but, if not, the default values for each one are the Sigmoid and
	 * the XavierUniform respectively.
	 * 
	 * The Layer is not meant to be used alone but, if you want to, you have to set
	 * the learningRate beforehand.
	 * 
	 * @param n_input    number of neurons in the previous layer
	 * @param n_neurons  numbers of neurons in the current layer
	 * @param activation activation function
	 * @param init       initializer
	 */
	public Layer(int n_input, int n_neurons, ActivationFunction activation, Initializer init) {
		if (n_neurons <= 0)
			throw new IllegalArgumentException("n_neurons can't be <= 0.");
		if (n_input <= 0)
			throw new IllegalArgumentException("n_input can't be <= 0.");

		this.n_neurons = n_neurons;
		this.n_input = n_input;

		this.activations = new float[n_neurons];

		this.activationFunction = activation;
		this.initializer = init;

		/*
		 * The aim of weight initialization is to prevent layer activation outputs from
		 * exploding or vanishing during the course of a forward pass through. If either
		 * occurs, loss gradients will either be too large or too small to flow
		 * backwards beneficially, and the network will take longer to converge, if it
		 * is even able to do so.
		 */
		this.weights = new float[n_neurons][n_input];
		init.initWeights(weights);

		this.biases = new float[n_neurons];
		init.initBiases(biases);

		delta_weights = new float[n_neurons][n_input];
		delta_biases = new float[n_neurons];

		/*
		 * The layer has no idea of the batch size, which is maintained at the network
		 * level. What it can do is keeping an internal counter to determine determine
		 * how many cycles have elapsed from the past update.
		 */
		batch = 0;

		learningRate = 0.5f;
	}

	/**
	 * Builds a layer with a XavierUniform initializer.
	 */
	public Layer(int n_input, int n_neurons, ActivationFunction activation) {
		this(n_input, n_neurons, activation, Initializer.XavierUniform);
	}

	/**
	 * Builds a layer with the Sigmoid as activation function.
	 */
	public Layer(int n_input, int n_neurons, Initializer init) {
		this(n_input, n_neurons, ActivationFunction.Sigmoid, init);
	}

	/**
	 * Builds a layer with a XavierUniform initializer and with the Sigmoid as
	 * activation function.
	 */
	public Layer(int n_input, int n_neurons) {
		this(n_input, n_neurons, ActivationFunction.Sigmoid, Initializer.XavierUniform);
	}

	// getters
	public ActivationFunction getActivationFunction() {
		return activationFunction;
	}

	public Initializer getInitializer() {
		return initializer;
	}

	public int getNeuronNumber() {
		return n_neurons;
	}

	public int getInputNumber() {
		return n_input;
	}

	public float[][] getWeights() {
		return weights;
	}

	public float[] getBiases() {
		return biases;
	}

	public float[] getOutput() {
		return activations;
	}

	// setters

	/**
	 * Mostly for debug purpose; lets you set the weights manually. An
	 * IllegalArgumentException is thrown if the size of the weights matrix passed
	 * as argument is not compliant with the size of the class weights matrix.
	 */
	public void setWeights(float[][] weights) {
		if (weights.length != this.weights.length)
			throw new IllegalArgumentException(
					String.format("this.weights.rows[%s] != weights.rows[%s].", this.weights.length, weights.length));
		if (weights[0].length != this.weights[0].length)
			throw new IllegalArgumentException(String.format("this.weights.cols[%s] != weights.cols[%s].",
					this.weights[0].length, weights[0].length));
		this.weights = weights;
	}

	/**
	 * Mostly for debug purpose; lets you set the biases manually. An
	 * IllegalArgumentException is thrown if the size of the biases vector passed as
	 * argument is not compliant with the class one.
	 */
	public void setBiases(float[] biases) {
		if (biases.length != this.biases.length)
			throw new IllegalArgumentException(
					String.format("this.biases.length[%s] != biases.length[%s].", this.biases.length, biases.length));
		this.biases = biases;
	}

	/**
	 * Sets the learning rate. The default one is 0.5f.
	 */
	public void setLearningRate(float learningRate) {
		this.learningRate = learningRate;
	}

	/**
	 * Feed the input through the layer. An IllegalArgumentException is thrown if
	 * the size of the input vector does not match with n_input.
	 */
	public float[] activate(float[] input) {
		if (input.length != n_input)
			throw new IllegalArgumentException(String.format("input.length[%s] != n_input[%s]", input.length, n_input));
		for (int i = 0; i < weights.length; i++)
			activations[i] = activationFunction.execute(MathUtils.dot(weights[i], input) + biases[i]);
		return activations;
	}

	/*
	 * Calculating the layer error must be done outside the Layer class itself
	 * because we need to have informations about the next layer in the network.
	 * Once calculated, the delta is stored inside the layer itself. We only need to
	 * store the deltas and not to update the weights and biases directly as the
	 * update depends on the batch size.
	 * 
	 * Only the Network object can modify the Layer.
	 */
	protected void addWeightsAndBiases(float[][] delta_weights, float[] delta_biases) {
		for (int i = 0; i < n_neurons; i++) {
			this.delta_biases[i] += delta_biases[i];
			for (int j = 0; j < n_input; j++)
				this.delta_weights[i][j] += delta_weights[i][j];
		}
		batch++;
	}

	/*
	 * Adjusts weights and biases.
	 */
	protected void adjustWeightsAndBiases() {
		adjustWeights();
		adjustBiases();
		resetDelta();
	}

	private void adjustWeights() {
		for (int i = 0; i < n_neurons; i++)
			for (int j = 0; j < n_input; j++)
				weights[i][j] -= (delta_weights[i][j] * learningRate) / batch;
	}

	private void adjustBiases() {
		for (int i = 0; i < n_neurons; i++)
			biases[i] -= (delta_biases[i] * learningRate) / batch;
	}

	private void resetDelta() {
		delta_weights = new float[n_neurons][n_input];
		delta_biases = new float[n_neurons];
		batch = 0;
	}

	private void readObject(ObjectInputStream aInputStream) throws ClassNotFoundException, IOException {

		// perform the default de-serialization first
		aInputStream.defaultReadObject();

		this.activations = new float[n_neurons];
		this.delta_weights = new float[n_neurons][n_input];
		this.delta_biases = new float[n_neurons];
	}
}
