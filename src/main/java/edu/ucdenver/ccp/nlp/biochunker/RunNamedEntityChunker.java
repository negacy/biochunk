package edu.ucdenver.ccp.nlp.biochunker;

import java.io.File;
import java.util.Arrays;

import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.collection.CollectionReader;
import org.apache.uima.jcas.JCas;
import org.cleartk.ml.CleartkSequenceAnnotator;
import org.cleartk.ml.jar.GenericJarClassifierFactory;
import org.cleartk.ml.jar.JarClassifierBuilder;
import org.cleartk.ne.type.NamedEntityMention;
import org.cleartk.syntax.opennlp.PosTaggerAnnotator;
import org.cleartk.syntax.opennlp.SentenceAnnotator;
import org.cleartk.token.tokenizer.TokenAnnotator;
import org.cleartk.util.ae.UriToDocumentTextAnnotator;
import org.cleartk.util.cr.UriCollectionReader;
import org.apache.uima.fit.component.JCasAnnotator_ImplBase;
import org.apache.uima.fit.factory.AggregateBuilder;
import org.apache.uima.fit.factory.AnalysisEngineFactory;
import org.apache.uima.fit.pipeline.SimplePipeline;
import org.apache.uima.fit.util.JCasUtil;

import com.lexicalscope.jewel.cli.CliFactory;
import com.lexicalscope.jewel.cli.Option;

/**
 * This class provides a main method that demonstrates how to run a trained
 * {@link NamedEntityChunker} on new files.
 * 
 * <br>
 * Copyright (c) 2012, Regents of the University of Colorado <br>
 * All rights reserved.
 * 
 * @author Steven Bethard
 */
public class RunNamedEntityChunker {

  public interface Options {
    @Option(
        longName = "model-dir",
        description = "The directory where the model was trained",
        defaultValue = "target/chunking/ne-model")
    public File getModelDirectory();

    @Option(
        longName = "text-file",
        description = "The file to label with named entities.",
        defaultValue = "/Users/negacy/workspace/bio-chunker/data/sample/2008_Sichuan_earthquake.txt")
    public File getTextFile();
  }

  public static void main(String[] args) throws Exception {
    Options options = CliFactory.parseArguments(Options.class, args);

    // a reader that loads the URIs of the text file
    CollectionReader reader = UriCollectionReader.getCollectionReaderFromFiles(Arrays.asList(options.getTextFile()));

    // assemble the classification pipeline
    AggregateBuilder aggregate = new AggregateBuilder();

    // an annotator that loads the text from the training file URIs
    aggregate.add(UriToDocumentTextAnnotator.getDescription());

    // annotators that identify sentences, tokens and part-of-speech tags in the text
    aggregate.add(SentenceAnnotator.getDescription());
    aggregate.add(TokenAnnotator.getDescription());
    aggregate.add(PosTaggerAnnotator.getDescription());

    // our NamedEntityChunker annotator, configured to classify on the new texts
    aggregate.add(AnalysisEngineFactory.createEngineDescription(
        NamedEntityChunker.class,
        CleartkSequenceAnnotator.PARAM_IS_TRAINING,
        false,
        GenericJarClassifierFactory.PARAM_CLASSIFIER_JAR_PATH,
        JarClassifierBuilder.getModelJarFile(options.getModelDirectory())));

    // a very simple annotator that just prints out any named entities we found
    aggregate.add(AnalysisEngineFactory.createEngineDescription(PrintNamedEntityMentions.class));

    // run the classification pipeline on the new texts
    SimplePipeline.runPipeline(reader, aggregate.createAggregateDescription());
  }

  /**
   * A simple annotator that just prints out any {@link NamedEntityMention}s in the CAS.
   * 
   * A real pipeline would probably decide on an appropriate output format and write files instead
   * of printing to standard output.
   */
  public static class PrintNamedEntityMentions extends JCasAnnotator_ImplBase {

    @Override
    public void process(JCas jCas) throws AnalysisEngineProcessException {
      for (NamedEntityMention mention : JCasUtil.select(jCas, NamedEntityMention.class)) {
        System.out.printf("%s (%s)\n", mention.getCoveredText(), mention.getMentionType());
      }
    }

  }
}