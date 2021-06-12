package main.training;

/**
 * Each Match is made of a data part and a label: this way it can be used both
 * to create datasets for training and to create datasets for network use
 * (containing only data, without labels). If you need to build a dataset for
 * network use just ignore the label (which will be set to -1) using
 * the appropriate constructor.
 * 
 * This framework allows you to create networks for image processing, so the
 * input will be an image, turned into a float array (one for each neuron in the
 * input layer). The original image has a width and a height and we'd like to
 * keep them in order to print some output on the console. These fields can be
 * left blank and in this case we will assume that the image has the same number
 * of pixels for the width and for the height (if possible).
 * 
 * In a CIFAR-10 problem, for example, the 0-9 digits are the possible results;
 * for each set of possible results, only one element between them is the
 * correct answer. We could think of a solution as a vector of booleans in which
 * only one element is true and the others are false. For simplicity, we only
 * store the index of this element, so the label can be, without loss
 * of generality, only one integer.
 * 
 */
public class Match {

	public int width;
	public int height;

	private float[] input;
	private int label;

	public Match(int width, int height, float[] input, int label) {

		if (width < 0)
			throw new IllegalArgumentException(String.format("Width [%d] must be >= 0", width));

		if (height < 0)
			throw new IllegalArgumentException(String.format("Height [%d] must be >= 0", height));

		this.width = width;
		this.height = height;
		this.input = input;
		this.label = label;
	}

	public Match(int width, int height, float[] input) {
		this(width, height, input, -1);
	}

	// width and height must be > 0 if set
	public Match(float[] input, int label) {
		this(0, 0, input, label);
	}

	public Match(float[] input) {
		this(0, 0, input, -1);
	}

	// getters

	public float[] getInput() {
		return input;
	}

	public int getLabel() {
		return label;
	}

	public boolean hasLabel () {
		return label > -1;
	}

	public int getWidth() {
		return width;
	}

	public int getHeight() {
		return height;
	}

	// setters

	public void setInput(float[] input) {
		this.input = input;
	}

	public void setLabel(int label) {
		this.label = label;
	}

	public void setWidth(int width) {
		this.width = width;
	}

	public void setHeight(int height) {
		this.height = height;
	}

	/*
	 * Converts a gray-scale pixel to an ascii-shade.
	 */
	private char toChar(float val) {
		return " .:-=+*#%@".charAt(min((int) (val * 10), 9));
	}

	private static int min(int a, int b) {
		return a < b ? a : b;
	}

	@Override
	public String toString() {
		final StringBuilder sb = new StringBuilder();
		for (int i = 0; i < width; i++) {
			for (int j = 0; j < height; j++)
				sb.append(toChar(input[i * width + j]));
			sb.append('\n');
		}
		return sb.toString();
	}
}
