package edu.ucdenver.ccp.nlp.biochunker;

import java.io.File;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.filefilter.FileFilterUtils;
import org.apache.uima.cas.CAS;
import org.apache.uima.collection.CollectionReader;
import org.apache.uima.fit.component.ViewCreatorAnnotator;
import org.apache.uima.fit.factory.AggregateBuilder;
import org.apache.uima.fit.factory.AnalysisEngineFactory;
import org.apache.uima.fit.factory.CollectionReaderFactory;
import org.apache.uima.fit.pipeline.JCasIterator;
import org.apache.uima.fit.pipeline.SimplePipeline;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.cleartk.ml.CleartkSequenceAnnotator;
import org.cleartk.ml.jar.DefaultSequenceDataWriterFactory;
import org.cleartk.ml.jar.DirectoryDataWriterFactory;
import org.cleartk.ml.jar.GenericJarClassifierFactory;
import org.cleartk.ml.jar.JarClassifierBuilder;
import org.cleartk.ml.jar.Train;
import org.cleartk.ml.mallet.MalletCrfStringOutcomeDataWriter;
import org.cleartk.eval.AnnotationStatistics;
import org.cleartk.eval.Evaluation_ImplBase;

import edu.ucdenver.ccp.nlp.biochunker.TrainNamedEntityChunker.MascTextFileFilter; //org.cleartk.examples.chunking.TrainNamedEntityChunker.MascTextFileFilter;

import org.cleartk.examples.chunking.util.MASCGoldAnnotator;
import org.cleartk.ne.type.NamedEntityMention;
import org.cleartk.syntax.opennlp.PosTaggerAnnotator;
import org.cleartk.syntax.opennlp.SentenceAnnotator;
import org.cleartk.token.tokenizer.TokenAnnotator;
import org.cleartk.util.ae.UriToDocumentTextAnnotator;
import org.cleartk.util.cr.UriCollectionReader;

import com.google.common.base.Function;
import com.lexicalscope.jewel.cli.CliFactory;
import com.lexicalscope.jewel.cli.Option;

/**
 * <p>
 * This class can be used to train and test the named entity chunker. It can be used to either
 * perform 2-fold cross-validation, or training and testing on a holdout test set, or just to
 * evaluate/test a pre-trained model.
 * 
 * <br>
 * Copyright (c) 2012, Regents of the University of Colorado <br>
 * All rights reserved.
 * 
 * @author Himanshu Gahlot
 * @author Steven Bethard
 * 
 * 
 */
public class EvaluateNamedEntityChunker extends
    Evaluation_ImplBase<File, AnnotationStatistics<String>> {

  public interface Options {
    @Option(
        longName = "train-dir",
        description = "Specify the directory containing the training documents.  This is used for cross-validation and for training in a holdout set evaluator. "
            + "When we run this example we point to a directory containing training data from the MASC-1.0.3 corpus - i.e. a directory called 'MASC-1.0.3/data/written'",
        defaultValue = "/Users/negacy/workspace/bio-chunker/data/MASC-1.0.3/data/written")
    public File getTrainDirectory();

    @Option(
        longName = "models-dir",
        description = "specify the directory in which to write out the trained model files",
        defaultValue = "target/chunking/ne-model")
    public File getModelsDirectory();
  }

  public static void main(String[] args) throws Exception {
    Options options = CliFactory.parseArguments(Options.class, args);

    // find training files
    List<File> trainFiles = new ArrayList<File>(FileUtils.listFiles(
        options.getTrainDirectory(),
        new MascTextFileFilter(),
        FileFilterUtils.falseFileFilter()));

    // run cross validation
    EvaluateNamedEntityChunker evaluator = new EvaluateNamedEntityChunker(
        options.getModelsDirectory());
    List<AnnotationStatistics<String>> foldStats = evaluator.crossValidation(trainFiles, 2);
    AnnotationStatistics<String> crossValidationStats = AnnotationStatistics.addAll(foldStats);

    System.err.println("Cross Validation Results:");
    System.err.print(crossValidationStats);
    System.err.println();
    System.err.println(crossValidationStats.confusions());
    System.err.println();
    
 
    // train and save a model using all the data
    evaluator.trainAndTest(trainFiles, Collections.<File> emptyList());
  }

  public EvaluateNamedEntityChunker(File baseDirectory) {
    super(baseDirectory);
  }

  @Override
  protected CollectionReader getCollectionReader(List<File> files) throws Exception {
    return CollectionReaderFactory.createReader(UriCollectionReader.getDescriptionFromFiles(files));
  }

  @Override
  public void train(CollectionReader collectionReader, File outputDirectory) throws Exception {
    // assemble the training pipeline
    AggregateBuilder aggregate = new AggregateBuilder();

    // an annotator that loads the text from the training file URIs
    aggregate.add(UriToDocumentTextAnnotator.getDescription());

    // an annotator that parses and loads MASC named entity annotations (and tokens)
    aggregate.add(MASCGoldAnnotator.getDescription());

    // an annotator that adds part-of-speech tags
    aggregate.add(PosTaggerAnnotator.getDescription());

    // our NamedEntityChunker annotator, configured to write Mallet CRF training data
    aggregate.add(AnalysisEngineFactory.createEngineDescription(
        NamedEntityChunker.class,
        CleartkSequenceAnnotator.PARAM_IS_TRAINING,
        true,
        DirectoryDataWriterFactory.PARAM_OUTPUT_DIRECTORY,
        outputDirectory,
        DefaultSequenceDataWriterFactory.PARAM_DATA_WRITER_CLASS_NAME,
        MalletCrfStringOutcomeDataWriter.class));

    // run the pipeline over the training corpus
    SimplePipeline.runPipeline(collectionReader, aggregate.createAggregateDescription());

    // quiet Mallet down a bit (but still leave likelihoods so you can see progress)
    Logger malletLogger = Logger.getLogger("cc.mallet");
    malletLogger.setLevel(Level.WARNING);
    Logger likelihoodLogger = Logger.getLogger("cc.mallet.fst.CRFOptimizableByLabelLikelihood");
    likelihoodLogger.setLevel(Level.INFO);

    // train a Mallet CRF model on the training data
    Train.main(outputDirectory);

  }

  @Override
  protected AnnotationStatistics<String> test(CollectionReader collectionReader, File modelDirectory)
      throws Exception {

    final String defaultViewName = CAS.NAME_DEFAULT_SOFA;
    final String goldViewName = "GoldView";

    // define the pipeline
    AggregateBuilder aggregate = new AggregateBuilder();

    // Annotators processing the gold view:
    // * create the gold view
    // * load the text
    // * load the MASC annotations
    aggregate.add(AnalysisEngineFactory.createEngineDescription(
        ViewCreatorAnnotator.class,
        ViewCreatorAnnotator.PARAM_VIEW_NAME,
        goldViewName));
    aggregate.add(UriToDocumentTextAnnotator.getDescription(), defaultViewName, goldViewName);
    aggregate.add(MASCGoldAnnotator.getDescription(), defaultViewName, goldViewName);

    // Annotators processing the default (system) view:
    // * load the text
    // * parse sentences, tokens, part-of-speech tags
    // * run the named entity chunker
    aggregate.add(UriToDocumentTextAnnotator.getDescription());
    aggregate.add(SentenceAnnotator.getDescription());
    aggregate.add(TokenAnnotator.getDescription());
    aggregate.add(PosTaggerAnnotator.getDescription());
    aggregate.add(AnalysisEngineFactory.createEngineDescription(
        NamedEntityChunker.class,
        CleartkSequenceAnnotator.PARAM_IS_TRAINING,
        false,
        GenericJarClassifierFactory.PARAM_CLASSIFIER_JAR_PATH,
        JarClassifierBuilder.getModelJarFile(modelDirectory)));

    // prepare the evaluation statistics
    AnnotationStatistics<String> stats = new AnnotationStatistics<String>();
    Function<NamedEntityMention, ?> getSpan = AnnotationStatistics.annotationToSpan();
    Function<NamedEntityMention, String> getCategory = AnnotationStatistics.annotationToFeatureValue("mentionType");

    // iterate over each JCas to be evaluated
    JCasIterator iter = new JCasIterator(collectionReader, aggregate.createAggregate());
    while (iter.hasNext()) {
      JCas jCas = iter.next();
      JCas goldView = jCas.getView(goldViewName);
      JCas systemView = jCas.getView(defaultViewName);

      // extract the named entity mentions from both gold and system views
      Collection<NamedEntityMention> goldMentions, systemMentions;
      goldMentions = JCasUtil.select(goldView, NamedEntityMention.class);
      systemMentions = JCasUtil.select(systemView, NamedEntityMention.class);

      // compare the system mentions to the gold mentions
      stats.add(goldMentions, systemMentions, getSpan, getCategory);
    }

    return stats;
  }
}
 
