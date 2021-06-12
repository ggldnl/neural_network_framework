package main.training;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

/**
 * Dataset object. Each dataset consists of at least one Match object.
 * 
 * Given a particular task, you need to extend the Dataset class to fit your
 * needs, as shown in the MNIST and EMNIST examples.
 */
public abstract class DataSet implements Iterable<Match> {

	private List<Match> matches;

	public DataSet() {
		matches = new ArrayList<>();
	}

	public void addMatch(Match match) {
		matches.add(match);
	}

	@Override
	public Iterator<Match> iterator() {
		return matches.iterator();
	}

	public int size() {
		return matches.size();
	}

	/**
	 * Shuffles the dataset.
	 */
	public void shuffle() {
		Collections.shuffle(matches);
	}

	/**
	 * Method to create a dataset to train the network. It's supposed that the
	 * dataset information is split between a data file and a label file.
	 * 
	 * @param dataFile	file containing dataset images
	 * @param labelFile	file containing the labels
	 */
	public abstract DataSet createSet(File dataFile, File labelFile);

	/**
	 * Method to create a dataset to test the network. We may not have a label file.
	 * 
	 * @param dataFile	file containing dataset images
	 */
	public abstract DataSet createSet(File dataFile);

	/**
	 * Method to create a dataset of n matches. If the file contains less than n entries
	 * an exception is thrown.
	 * 
	 * @param dataFile	file containing dataset images
	 * @param labelFile	file containing the labels
	 * @param number of image-label matches to put in the dataset
	 */
	public abstract DataSet createSet(File dataFile, File labelFile, int matches);
	
	public abstract DataSet createSet(File dataFile, int matches);
}