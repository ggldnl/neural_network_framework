package main.test.emnist;

import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

import main.training.DataSet;
import main.training.Match;

/**
 * This class incorporates the logic for a Dataset object creation starting from
 * the binary files containing the EMNIST dataset.
 * 
 * https://www.nist.gov/itl/products-and-services/emnist-dataset
 */
public class LetterDataSet extends DataSet {
	
	private static final int IMG_FILE_MAGIC_INT = 2051;
	private static final int LABEL_FILE_MAGIC_INT = 2049;

	@Override
	public DataSet createSet (File dataFile, File labelFile) {
		
		DataSet set = new LetterDataSet();

		try (DataInputStream image_stream = new DataInputStream(new FileInputStream(dataFile));
				DataInputStream label_stream = new DataInputStream(new FileInputStream(labelFile));) {

			if (image_stream.readInt() != IMG_FILE_MAGIC_INT)
				throw new IOException(String.format("Unknown file format for: %s.", dataFile.getName()));

			if (label_stream.readInt() != LABEL_FILE_MAGIC_INT)
				throw new IOException(String.format("Unknown file format for: %s.", labelFile.getName()));

			int nImages = image_stream.readInt();
			int nLabels = label_stream.readInt();

			if (nImages != nLabels)
				throw new IOException(
						String.format("File %s and file %s contains data for a different number of images.",
								dataFile.getName(), labelFile.getName()));

			int rows = image_stream.readInt();
			int cols = image_stream.readInt();

			for (int i = 0; i < nImages; i++) {
				
				byte[] data = new byte[rows * cols];
				byte[][] formatted = new byte [rows][cols];
				
				image_stream.read(data, 0, data.length);
				
				for (int r = 0; r < rows; r++)
					for (int c = 0; c < cols; c++)
						formatted[r][c] = data[r * rows + c];
				
				/*
				 * images are 90 degrees clockwise
				 */
				float [] img = rotate (formatted);

				/*
				 * 
				 * 	0	1	2
				 * 	3	4	5
				 * 	6	7	8
				 * 
				 * 0, 1, 2, 3, 4, 5, 6, 7, 8 
				 * 
				 * 		|
				 * 		v
				 * 
				 * 	0	3	6
				 * 	1	4	7
				 * 	2	5	8
				 * 
				 * 0, 3, 6, 1, 4, 7, 2, 5, 8
				 */
				
				/*
				 * a -> 0
				 * b -> 1
				 * and so on...
				 */
				int label = label_stream.readByte() - 1; // a in the ascii table is #97
				
				set.addMatch(new Match(rows, cols, img, label));
			}

		} catch (Exception e) {
			e.printStackTrace();
		}

		return set;
	}
	
	@Override
	public DataSet createSet (File dataFile, File labelFile, int matches) {
		
		if (matches < 0)
			throw new IllegalArgumentException ("The number of matches in the dataset must be positive.");
		
		DataSet set = new LetterDataSet();

		try (DataInputStream image_stream = new DataInputStream(new FileInputStream(dataFile));
				DataInputStream label_stream = new DataInputStream(new FileInputStream(labelFile));) {

			if (image_stream.readInt() != IMG_FILE_MAGIC_INT)
				throw new IOException(String.format("Unknown file format for: %s.", dataFile.getName()));

			if (label_stream.readInt() != LABEL_FILE_MAGIC_INT)
				throw new IOException(String.format("Unknown file format for: %s.", labelFile.getName()));

			int nImages = image_stream.readInt();
			int nLabels = label_stream.readInt();

			if (nImages != nLabels)
				throw new IOException(
						String.format("File %s and file %s contains data for a different number of images.",
								dataFile.getName(), labelFile.getName()));

			if (nImages < matches)
				throw new IllegalArgumentException("The number of matches given exceeds the number of elements in the file.");
			
			int rows = image_stream.readInt();
			int cols = image_stream.readInt();

			for (int i = 0; i < nImages; i++) {
				
				byte[] data = new byte[rows * cols];
				byte[][] formatted = new byte [rows][cols];
				
				image_stream.read(data, 0, data.length);
				
				for (int r = 0; r < rows; r++)
					for (int c = 0; c < cols; c++)
						formatted[r][c] = data[r * rows + c];
				
				/*
				 * images are 90 degrees clockwise
				 */
				float [] img = rotate (formatted);

				/*
				 * 
				 * 	0	1	2
				 * 	3	4	5
				 * 	6	7	8
				 * 
				 * 0, 1, 2, 3, 4, 5, 6, 7, 8 
				 * 
				 * 		|
				 * 		v
				 * 
				 * 	0	3	6
				 * 	1	4	7
				 * 	2	5	8
				 * 
				 * 0, 3, 6, 1, 4, 7, 2, 5, 8
				 */
				
				/*
				 * a -> 0
				 * b -> 1
				 * and so on...
				 */
				int label = label_stream.readByte() - 1; // a in the ascii table is #97
				
				set.addMatch(new Match(rows, cols, img, label));
			}

		} catch (Exception e) {
			e.printStackTrace();
		}

		return set;
	}
	
	private static float [] rotate (byte [][] array) {
		float [] res = new float [array.length * array[0].length];

		for (int i = 0; i < array.length; i++)
			for (int j = 0; j < array[i].length; j++) {
				res [j * array[0].length + i] = (array[i][j] & 255) / 255.0f;
			}

		return res;
	}
	
	@Override
	public DataSet createSet (File dataFile) {
		
		DataSet set = new LetterDataSet();

		try (DataInputStream image_stream = new DataInputStream(new FileInputStream(dataFile));) {

			if (image_stream.readInt() != IMG_FILE_MAGIC_INT)
				throw new IOException(String.format("Unknown file format for: %s.", dataFile.getName()));

			int nImages = image_stream.readInt();

			int rows = image_stream.readInt();
			int cols = image_stream.readInt();

			for (int i = 0; i < nImages; i++) {
				
				byte[] data = new byte[rows * cols];
				byte[][] formatted = new byte [rows][cols];
				
				image_stream.read(data, 0, data.length);
				
				for (int r = 0; r < rows; r++)
					for (int c = 0; c < cols; c++)
						formatted[r][c] = data[r * rows + c];
				
				/*
				 * images are 90 degrees clockwise
				 */
				float [] img = rotate (formatted);

				/*
				 * 
				 * 	0	1	2
				 * 	3	4	5
				 * 	6	7	8
				 * 
				 * 0, 1, 2, 3, 4, 5, 6, 7, 8 
				 * 
				 * 		|
				 * 		v
				 * 
				 * 	0	3	6
				 * 	1	4	7
				 * 	2	5	8
				 * 
				 * 0, 3, 6, 1, 4, 7, 2, 5, 8
				 */
				
				set.addMatch(new Match(rows, cols, img));
			}

		} catch (Exception e) {
			e.printStackTrace();
		}

		return set;
	}
	
	@Override
	public DataSet createSet (File dataFile, int matches) {
		
		if (matches < 0)
			throw new IllegalArgumentException ("The number of matches in the dataset must be positive.");
		
		DataSet set = new LetterDataSet();

		try (DataInputStream image_stream = new DataInputStream(new FileInputStream(dataFile));) {

			if (image_stream.readInt() != IMG_FILE_MAGIC_INT)
				throw new IOException(String.format("Unknown file format for: %s.", dataFile.getName()));

			int nImages = image_stream.readInt();

			if (nImages < matches)
				throw new IllegalArgumentException("The number of matches given exceeds the number of elements in the file.");
			
			int rows = image_stream.readInt();
			int cols = image_stream.readInt();

			for (int i = 0; i < nImages; i++) {
				
				byte[] data = new byte[rows * cols];
				byte[][] formatted = new byte [rows][cols];
				
				image_stream.read(data, 0, data.length);
				
				for (int r = 0; r < rows; r++)
					for (int c = 0; c < cols; c++)
						formatted[r][c] = data[r * rows + c];
				
				/*
				 * images are 90 degrees clockwise
				 */
				float [] img = rotate (formatted);

				/*
				 * 
				 * 	0	1	2
				 * 	3	4	5
				 * 	6	7	8
				 * 
				 * 0, 1, 2, 3, 4, 5, 6, 7, 8 
				 * 
				 * 		|
				 * 		v
				 * 
				 * 	0	3	6
				 * 	1	4	7
				 * 	2	5	8
				 * 
				 * 0, 3, 6, 1, 4, 7, 2, 5, 8
				 */
				
				set.addMatch(new Match(rows, cols, img));
			}

		} catch (Exception e) {
			e.printStackTrace();
		}

		return set;
	}
	
	/*
	 * Show the dataset
	 */
	public static void main (String ...strings) throws Exception {
		
		File dataFile = new File(LetterDataSet.class.getResource("/resources/emnist/emnist-letters-train-images-idx3-ubyte").getPath());
		File labelFile = new File(LetterDataSet.class.getResource("/resources/emnist/emnist-letters-train-labels-idx1-ubyte").getPath());
		DataSet dataSet = new LetterDataSet().createSet(dataFile, labelFile);
	
		for (Match match : dataSet) {
			int label = match.getLabel();
			System.out.println("Label: " + label + "\tChar: " + (char) (label + 96));
			System.out.println(match);
			System.out.println();
			Thread.sleep(1000);
		}
	}
}
