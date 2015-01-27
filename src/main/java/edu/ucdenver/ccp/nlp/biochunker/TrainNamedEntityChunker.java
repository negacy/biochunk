package edu.ucdenver.ccp.nlp.biochunker;


import java.io.File;

import org.apache.commons.io.filefilter.IOFileFilter;
import org.apache.uima.collection.CollectionReaderDescription;
import org.cleartk.ml.CleartkSequenceAnnotator;
import org.cleartk.ml.jar.DefaultSequenceDataWriterFactory;
import org.cleartk.ml.jar.DirectoryDataWriterFactory;
import org.cleartk.ml.jar.Train;

import org.cleartk.ml.mallet.MalletCrfStringOutcomeDataWriter;
 

//import org.cleartk.examples.chunking.util.MascGoldAnnotator;
import org.cleartk.examples.chunking.util.MASCGoldAnnotator;
import org.cleartk.syntax.opennlp.PosTaggerAnnotator;

import org.cleartk.util.ae.UriToDocumentTextAnnotator;
import org.cleartk.util.cr.UriCollectionReader;
import org.apache.uima.fit.factory.AggregateBuilder;
import org.apache.uima.fit.factory.AnalysisEngineFactory;
import org.apache.uima.fit.pipeline.SimplePipeline;



import com.lexicalscope.jewel.cli.CliFactory;
import com.lexicalscope.jewel.cli.Option;

/**
 * This class provides a main method that demonstrates how to train a {@link NamedEntityChunker} on
 * an annotated corpus of named entities.
 * 
 * <br>
 * Copyright (c) 2012, Regents of the University of Colorado <br>
 * All rights reserved.
 * 
 * @author Steven Bethard
 */
public class TrainNamedEntityChunker {

  public interface Options {
    @Option(
        longName = "train-dir",
        description = "The directory containing MASC-annotated files",
        defaultValue = "/Users/negacy/workspace/bio-chunker/data/MASC-1.0.3/data/written")
    public File getTrainDirectory();

    @Option(
        longName = "model-dir",
        description = "The directory where the model should be written",
        defaultValue = "target/chunking/ne-model")
    public File getModelDirectory();
  }

  public static void main(String[] args) throws Exception {
    Options options = CliFactory.parseArguments(Options.class, args);
    
     
    // a reader that loads the URIs of the training files
    CollectionReaderDescription reader = UriCollectionReader.getDescriptionFromDirectory(
        options.getTrainDirectory(),
        MascTextFileFilter.class,
        null);
    
    // assemble the training pipeline
    AggregateBuilder aggregate = new AggregateBuilder();

    // an annotator that loads the text from the training file URIs
    aggregate.add(UriToDocumentTextAnnotator.getDescription());
    
     
    // an annotator that parses and loads MASC named entity annotations (and tokens)
    aggregate.add(MASCGoldAnnotator.getDescription());
     
    // an annotator that adds part-of-speech tags (so we can use them for features)
    aggregate.add(PosTaggerAnnotator.getDescription());
     
    // our NamedEntityChunker annotator, configured to write Mallet CRF training data
    aggregate.add(AnalysisEngineFactory.createEngineDescription(
        NamedEntityChunker.class,
        CleartkSequenceAnnotator.PARAM_IS_TRAINING,
        true,
        DirectoryDataWriterFactory.PARAM_OUTPUT_DIRECTORY,
        options.getModelDirectory(),
        DefaultSequenceDataWriterFactory.PARAM_DATA_WRITER_CLASS_NAME,
        MalletCrfStringOutcomeDataWriter.class));
     
    // run the pipeline over the training corpus
    SimplePipeline.runPipeline(reader, aggregate.createAggregateDescription());
     
    // train a Mallet CRF model on the training data
    Train.main(options.getModelDirectory());
    System.out.println("... training done ...");
    
  }

  /**
   * An auxiliary class necessary to only load the ".txt" files from the MASC directories.
   * 
   * You can mostly ignore this - it's only necessary due to the idiosyncracies of the MASC
   * directory structure.
   */
  public static class MascTextFileFilter implements IOFileFilter {
    //@Override
    public boolean accept(File file) {
      return file.getPath().endsWith(".txt");
    }

    //@Override
    public boolean accept(File dir, String name) {
      return name.endsWith(".txt");
    }
  }

}