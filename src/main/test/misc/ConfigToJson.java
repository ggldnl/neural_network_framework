package main.test.misc;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;

import javax.swing.JFileChooser;
import javax.swing.UIManager;
import javax.swing.filechooser.FileFilter;

import com.google.gson.Gson;

import main.Network;

public class ConfigToJson {

	private static class RestoreFileFilter extends FileFilter {

		public boolean accept(File file) {
			if (file.isDirectory())
				return true;
			String fname = file.getName().toLowerCase();
			return fname.endsWith(Network.ext);
		}

		public String getDescription() {
			return "Serializable Network File";
		}
	}

	public static void main(String... strings) throws IOException {

		try {
			UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
		} catch (Exception e) {
			e.printStackTrace();
		}

		JFileChooser fileChooser = new JFileChooser();
		fileChooser.setDialogTitle("Select ." + Network.ext + " file to restore");
		fileChooser.setFileFilter(new RestoreFileFilter());
		int n = fileChooser.showOpenDialog(null);

		if (n == JFileChooser.APPROVE_OPTION) {

			// restore the network
			File restore = fileChooser.getSelectedFile();
			Network network = Network.restoreNetwork(restore);
			
			fileChooser.setDialogTitle("Select a file to save the JSON");
			fileChooser.resetChoosableFileFilters();
			n = fileChooser.showSaveDialog(null);

			if (n == JFileChooser.APPROVE_OPTION) {
				String filePath = fileChooser.getSelectedFile().getAbsolutePath();

				// write the json on a file
				Gson gson = new Gson();
				Writer writer = new FileWriter(filePath);
				gson.toJson(network, writer);
				writer.flush();
		        writer.close();
			}
		}
	}
}
