package com.machinelearning.NaiveBayes;

/*
 * Authors : Aniket Bhosale and Mayur Tare
 * Description : This class implements Naive Bayes algorithm with Laplace correction.
 */

import java.util.ArrayList;
import java.util.HashMap;

public class NaiveBayes {
	public static ArrayList<Double> weights;
	//Assume class label "1" to be true and rest to be false
	public static String trueClassLable = "1";
	public static double accuracy = 0.0;
	public static int c = 0;
	public static int w = 0;
	
	
	public static void main(String[] args) {
		
		//Read the file name for training data from config file
		String trainFilePath = Config.readConfig("trainFileName");
		String testFilePath = Config.readConfig("testFileName");
		
		//Read Training Examples from Dataset file
		ArrayList<Example> allexamples = DataLoader.readRecords(trainFilePath);
		ArrayList<Example> trainExamples = DataLoader.readRecords(trainFilePath);
		
		ArrayList<ArrayList<Example>> lists;
		
		int numberofFolds = 10;
		/***********************************************/
		for(int x=0; x<numberofFolds; x++){
			ArrayList<Example> examples = new ArrayList<>();
//			trainExamples = (ArrayList<Example>) allexamples.clone();
//			ArrayList<Example> foldTest = new ArrayList<>();
			
			lists = nFoldData.getFolds(x, examples, numberofFolds);
			
			examples = (ArrayList<Example>) lists.get(0).clone();
//			String[] ind = indices.split(",");
//			int startInd = Integer.parseInt(ind[0]);
//			int endInd = Integer.parseInt(ind[1]);
//			System.out.println(startInd+" : "+endInd);
			
//			for(int m = startInd; m<=endInd; m++){
//				foldTest.add(trainExamples.get(m));
//			}
//			
//			for(int m = 0; m<startInd; m++){
//				examples.add(trainExamples.get(m));
//			}
//			for(int n = endInd+1; n<trainExamples.size(); n++){
//				examples.add(trainExamples.get(n));
//			}
		
		/***********************************************/
		
		System.out.println("Examples =" + examples.size());
		
		//Data Structure to store the counts for features
		HashMap<Integer, HashMap<String, Integer>> trueCounts = new HashMap<Integer, HashMap<String, Integer>>();
		HashMap<Integer, HashMap<String, Integer>> falseCounts = new HashMap<Integer, HashMap<String, Integer>>();
		
		//Index of the class label
		String lable = Config.readConfig("classLable");
		int lableIndex = DataLoader.labels.indexOf(lable);
		
		int trueLableExamples = 0;
		String currVal = null;
		
		//Iterate over all training examples and learn.
		for(Example ex : examples){
			//Count for true class lable
			if(ex.features.get(lableIndex).equalsIgnoreCase(trueClassLable)){
				trueLableExamples++;
				for(int i = 0; i < DataLoader.numberOfFeatures; i++){
					currVal = ex.features.get(i);
					
					if(!trueCounts.containsKey(i)){
						HashMap<String, Integer> featureMap = new HashMap<String, Integer>();
						featureMap.put(currVal, 1);
						trueCounts.put(i, featureMap);
					}
					else{
						if(!trueCounts.get(i).containsKey(currVal)){
							trueCounts.get(i).put(currVal, 1);
						}
						else{
							int currCount = trueCounts.get(i).get(currVal);
							trueCounts.get(i).put(currVal, currCount+1);
						}
					}					
				}
			}
			//Count for false class label
			else{
				for(int i = 0; i < DataLoader.numberOfFeatures; i++){
					currVal = ex.features.get(i);
					
					if(!falseCounts.containsKey(i)){
						HashMap<String, Integer> featureMap = new HashMap<String, Integer>();
						featureMap.put(currVal, 1);
						falseCounts.put(i, featureMap);
					}
					else{
						if(!falseCounts.get(i).containsKey(currVal)){
							falseCounts.get(i).put(currVal, 1);
						}
						else{
							int currCount = falseCounts.get(i).get(currVal);
							falseCounts.get(i).put(currVal, currCount+1);
						}
					}					
				}
			}
		}
		//Total number of False lable Examples
		int falseLableExamples = examples.size() - trueLableExamples;
		
		//Possible class lables
		int possClassLables = 2;

		
		//Classify the test data
		//Read Test examples from Test Dataset
		//ArrayList<Example> testExamples = DataLoader.readRecords(testFilePath);
		
		//fold test data
		ArrayList<Example> testExamples = (ArrayList<Example>) lists.get(1).clone();;
		
		System.out.println("Test Examples =" + testExamples.size());
		
		int wrongPredctionCount = 0;
		int correctPredctionCount = 0;
		int laplaceCorrection = 1;
		
		for(Example testEx : testExamples){
			String observedLable = testEx.getFeature(lableIndex);
			
			double trueClassPrior = (double)(trueLableExamples + laplaceCorrection) / (trueLableExamples + falseLableExamples + possClassLables);
			double falseClassPrior = (double)(falseLableExamples + laplaceCorrection) / (trueLableExamples + falseLableExamples + possClassLables);
			double trueProb = 1.0 * trueClassPrior;
			double falseProb = 1.0 * falseClassPrior;
			
			for(int i = 0; i < DataLoader.numberOfFeatures; i++){
				String featureVal = testEx.getFeature(i);
				int numberOfPossVals = DataLoader.featurePossVals.get(i).size();
				if(i != lableIndex){
					//for unseen values of features in train data assigning count = 0
					int featureTrueCount = 0;
					int featureFalseCount = 0;
					if(trueCounts != null && trueCounts.get(i) != null)
						featureTrueCount = trueCounts.get(i).get(featureVal) != null ? trueCounts.get(i).get(featureVal) : 0;
					if(falseCounts != null && falseCounts.get(i) != null)	
						featureFalseCount = falseCounts.get(i).get(featureVal) != null ? falseCounts.get(i).get(featureVal) : 0;
					
					trueProb *= (double)(featureTrueCount + laplaceCorrection)/(trueLableExamples + numberOfPossVals);
					falseProb *= (double)(featureFalseCount + laplaceCorrection)/(falseLableExamples + numberOfPossVals);
					
				}
			}
			
			double normTrueProb = trueProb / (trueProb + falseProb);
			double normFalseProb = falseProb / (trueProb + falseProb);
			
			
			String prediction = normTrueProb - normFalseProb > -0.3? "T" : "F";
			
			System.out.println("True Prob : "+normTrueProb+" false prob : "+normFalseProb+
					" Actual : "+observedLable+" predicted : "+prediction
					);
			
			//Check if prediction is correct or not
			if(observedLable.equalsIgnoreCase(trueClassLable) && !prediction.equalsIgnoreCase("T")){
				wrongPredctionCount++;
			}
			else if(!observedLable.equalsIgnoreCase(trueClassLable) && prediction.equalsIgnoreCase("T")){
				wrongPredctionCount++;				
			}
			else{
				correctPredctionCount++;
			}
		}
		
		//Print the report of the classification
		System.out.println(correctPredctionCount+" Correct Predictions for "+testExamples.size()+" test examples");
		System.out.println(wrongPredctionCount+" Incorrect Predictions for "+testExamples.size()+" test examples");
		
		c += correctPredctionCount;
		w += wrongPredctionCount;
		
		accuracy += 100.0*correctPredctionCount/testExamples.size();
		System.out.println("Accuracy = "+accuracy);
		
		}//End fold for loop
		System.out.println(c+"---"+w);
	}

}
