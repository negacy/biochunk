package edu.ucdenver.ccp.nlp.biochunker;
/*
import org.apache.uima.UimaContext;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.ResourceInitializationException;
import org.cleartk.ml.CleartkSequenceAnnotator;
import org.cleartk.ml.chunking.BioChunking;
import org.cleartk.ml.feature.extractor.CleartkExtractor;
import org.cleartk.ml.feature.extractor.FeatureExtractor1;
import org.cleartk.ne.type.NamedEntityMention;
import org.cleartk.token.type.Token;
import org.cleartk.ml.feature.extractor.*; //CharacterCategoryPatternExtractor;
//import org.cleartk.ml.feature.extractor.TypePathExtractor;
*/
import java.util.ArrayList;
import java.util.List;

import org.apache.uima.UimaContext;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.ResourceInitializationException;
import org.cleartk.ml.CleartkSequenceAnnotator;
import org.cleartk.ml.Feature;
import org.cleartk.ml.Instances;
import org.cleartk.ml.chunking.BioChunking;
import org.cleartk.ml.feature.extractor.CleartkExtractor;
import org.cleartk.ml.feature.extractor.CleartkExtractor.Following;
import org.cleartk.ml.feature.extractor.CleartkExtractor.Preceding;
import org.cleartk.ml.feature.extractor.CombinedExtractor1;
import org.cleartk.ml.feature.extractor.CoveredTextExtractor;
import org.cleartk.ml.feature.extractor.FeatureExtractor1;
import org.cleartk.ml.feature.extractor.TypePathExtractor;
import org.cleartk.ml.feature.function.CharacterCategoryPatternFunction;
import org.cleartk.ml.feature.function.CharacterCategoryPatternFunction.PatternType;
import org.cleartk.ml.feature.function.FeatureFunctionExtractor;
import org.cleartk.ne.type.NamedEntityMention;
import org.cleartk.token.type.Sentence;
import org.cleartk.token.type.Token;

import org.apache.uima.analysis_component.JCasAnnotator_ImplBase;

//import org.cleartk.ml.feature.extractor.CharacterCategoryPatternExtractor;
 
public class NamedEntityChunker extends CleartkSequenceAnnotator<String> {
 
	private FeatureExtractor1<Token> extractor;
	private CleartkExtractor<Token, Token> contextExtractor; 
	private BioChunking<Token, NamedEntityMention> chunking;
	
	  
	@Override
	  public void initialize(UimaContext context) throws ResourceInitializationException {
	    super.initialize(context);

	    // the token feature extractor: text, char pattern (uppercase, digits, etc.), and part-of-speech
	    this.extractor = new CombinedExtractor1<Token>(
	        new FeatureFunctionExtractor<Token>(
	            new CoveredTextExtractor<Token>(),
	            new CharacterCategoryPatternFunction<Token>(PatternType.REPEATS_MERGED)),
	        new TypePathExtractor<Token>(Token.class, "pos"));

	    // the context feature extractor: the features above for the 3 preceding and 3 following tokens
	    this.contextExtractor = new CleartkExtractor<Token, Token>(
	        Token.class,
	        this.extractor,
	        new Preceding(3),
	        new Following(3));

	    // the chunking definition: Tokens will be combined to form NamedEntityMentions, with labels
	    // from the "mentionType" attribute so that we get B-location, I-person, etc.
	    this.chunking = new BioChunking<Token, NamedEntityMention>(
	        Token.class,
	        NamedEntityMention.class,
	        "mentionType");
	  }

	  @Override
	  public void process(JCas jCas) throws AnalysisEngineProcessException {
	    for (Sentence sentence : JCasUtil.select(jCas, Sentence.class)) {

	      // extract features for each token in the sentence
	      List<Token> tokens = JCasUtil.selectCovered(jCas, Token.class, sentence);
	      List<List<Feature>> featureLists = new ArrayList<List<Feature>>();
	      for (Token token : tokens) {
	        List<Feature> features = new ArrayList<Feature>();
	        features.addAll(this.extractor.extract(jCas, token));
	        features.addAll(this.contextExtractor.extract(jCas, token));
	        featureLists.add(features);
	      }

	      // during training, convert NamedEntityMentions in the CAS into expected classifier outcomes
	      if (this.isTraining()) {

	        // extract the gold (human annotated) NamedEntityMention annotations
	        List<NamedEntityMention> namedEntityMentions = JCasUtil.selectCovered(
	            jCas,
	            NamedEntityMention.class,
	            sentence);

	        // convert the NamedEntityMention annotations into token-level BIO outcome labels
	        List<String> outcomes = this.chunking.createOutcomes(jCas, tokens, namedEntityMentions);

	        // write the features and outcomes as training instances
	        this.dataWriter.write(Instances.toInstances(outcomes, featureLists));
	      }

	      // during classification, convert classifier outcomes into NamedEntityMentions in the CAS
	      else {

	        // get the predicted BIO outcome labels from the classifier
	        List<String> outcomes = this.classifier.classify(featureLists);

	        // create the NamedEntityMention annotations in the CAS
	        this.chunking.createChunks(jCas, tokens, outcomes);
	      }
	    }
	  }
	    
	
}
