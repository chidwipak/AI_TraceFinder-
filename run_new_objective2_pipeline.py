#!/usr/bin/env python3
"""
New Objective 2: Complete Pipeline Runner
Runs the complete comprehensive tampered/original detection pipeline
"""

import os
import sys
import logging
import subprocess
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NewObjective2PipelineRunner:
    def __init__(self):
        self.pipeline_start_time = datetime.now()
        self.results_summary = {}
        
    def run_script(self, script_name, description):
        """Run a Python script and log results"""
        logger.info("="*80)
        logger.info(f"RUNNING: {description}")
        logger.info(f"Script: {script_name}")
        logger.info("="*80)
        
        try:
            result = subprocess.run([sys.executable, script_name], 
                                  capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                logger.info(f"✅ {description} completed successfully!")
                self.results_summary[script_name] = "SUCCESS"
                return True
            else:
                logger.error(f"❌ {description} failed!")
                logger.error(f"Error: {result.stderr}")
                self.results_summary[script_name] = f"FAILED: {result.stderr}"
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ {description} timed out after 1 hour!")
            self.results_summary[script_name] = "TIMEOUT"
            return False
        except Exception as e:
            logger.error(f"❌ {description} failed with exception: {e}")
            self.results_summary[script_name] = f"EXCEPTION: {e}"
            return False
    
    def check_dependencies(self):
        """Check if all required dependencies are available"""
        logger.info("🔍 CHECKING DEPENDENCIES")
        
        required_packages = [
            ('pandas', 'pandas'), ('numpy', 'numpy'), ('cv2', 'opencv-python'), 
            ('sklearn', 'scikit-learn'), ('matplotlib', 'matplotlib'), 
            ('seaborn', 'seaborn'), ('scipy', 'scipy'), ('skimage', 'scikit-image'), 
            ('pywt', 'PyWavelets'), ('mahotas', 'mahotas'),
            ('imblearn', 'imbalanced-learn'), ('albumentations', 'albumentations'), 
            ('pdf2image', 'pdf2image'), ('fitz', 'PyMuPDF')
        ]
        
        missing_packages = []
        
        for import_name, package_name in required_packages:
            try:
                __import__(import_name)
                logger.info(f"✅ {package_name}")
            except ImportError:
                missing_packages.append(package_name)
                logger.warning(f"❌ {package_name}")
        
        if missing_packages:
            logger.warning(f"Missing packages: {missing_packages}")
            logger.warning("Please install missing packages before running the pipeline")
            return False
        
        logger.info("✅ All dependencies are available!")
        return True
    
    def run_complete_pipeline(self):
        """Run the complete Objective 2 pipeline"""
        logger.info("🚀 STARTING NEW OBJECTIVE 2 COMPREHENSIVE PIPELINE")
        logger.info("="*80)
        logger.info(f"Pipeline started at: {self.pipeline_start_time}")
        logger.info("="*80)
        
        # Check dependencies first
        if not self.check_dependencies():
            logger.error("❌ Dependency check failed. Please install missing packages.")
            return False
        
        # Pipeline steps
        pipeline_steps = [
            {
                'script': 'new_objective2_data_analysis.py',
                'description': 'Data Analysis Phase - Convert PDFs to images and analyze all data sources'
            },
            {
                'script': 'new_objective2_data_preprocessing.py', 
                'description': 'Data Preprocessing - Create comprehensive train/test splits with class balancing'
            },
            {
                'script': 'new_objective2_feature_extraction.py',
                'description': 'Feature Extraction - Extract comprehensive forensic features using all available data'
            }
        ]
        
        # Run each step
        successful_steps = 0
        total_steps = len(pipeline_steps)
        
        for step in pipeline_steps:
            if self.run_script(step['script'], step['description']):
                successful_steps += 1
            else:
                logger.error(f"❌ Pipeline failed at step: {step['description']}")
                break
        
        # Generate final summary
        self.generate_pipeline_summary(successful_steps, total_steps)
        
        return successful_steps == total_steps
    
    def generate_pipeline_summary(self, successful_steps, total_steps):
        """Generate comprehensive pipeline summary"""
        pipeline_end_time = datetime.now()
        duration = pipeline_end_time - self.pipeline_start_time
        
        logger.info("="*80)
        logger.info("NEW OBJECTIVE 2 PIPELINE SUMMARY")
        logger.info("="*80)
        logger.info(f"Pipeline Duration: {duration}")
        logger.info(f"Successful Steps: {successful_steps}/{total_steps}")
        logger.info(f"Success Rate: {successful_steps/total_steps*100:.1f}%")
        
        logger.info("\nStep Results:")
        for script, result in self.results_summary.items():
            status_icon = "✅" if result == "SUCCESS" else "❌"
            logger.info(f"  {status_icon} {script}: {result}")
        
        if successful_steps == total_steps:
            logger.info("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("Next steps:")
            logger.info("1. Run baseline ML models training")
            logger.info("2. Run deep learning models training") 
            logger.info("3. Create ensemble models")
            logger.info("4. Evaluate final performance")
            logger.info("5. Analyze results and identify improvements for 90%+ accuracy")
        else:
            logger.error("\n❌ PIPELINE FAILED!")
            logger.error("Please check the logs and fix the issues before proceeding.")
        
        # Save summary to file
        summary_path = "new_objective2_pipeline_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("NEW OBJECTIVE 2 COMPREHENSIVE PIPELINE SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Pipeline Start Time: {self.pipeline_start_time}\n")
            f.write(f"Pipeline End Time: {pipeline_end_time}\n")
            f.write(f"Pipeline Duration: {duration}\n")
            f.write(f"Successful Steps: {successful_steps}/{total_steps}\n")
            f.write(f"Success Rate: {successful_steps/total_steps*100:.1f}%\n\n")
            
            f.write("Step Results:\n")
            f.write("-"*40 + "\n")
            for script, result in self.results_summary.items():
                f.write(f"{script}: {result}\n")
            
            if successful_steps == total_steps:
                f.write("\n✅ PIPELINE COMPLETED SUCCESSFULLY!\n")
                f.write("\nNext Steps:\n")
                f.write("1. Run baseline ML models training\n")
                f.write("2. Run deep learning models training\n")
                f.write("3. Create ensemble models\n")
                f.write("4. Evaluate final performance\n")
                f.write("5. Analyze results and identify improvements for 90%+ accuracy\n")
            else:
                f.write("\n❌ PIPELINE FAILED!\n")
                f.write("Please check the logs and fix the issues before proceeding.\n")
        
        logger.info(f"Pipeline summary saved to: {summary_path}")

def main():
    """Main function to run the complete pipeline"""
    runner = NewObjective2PipelineRunner()
    success = runner.run_complete_pipeline()
    
    if success:
        print("\n🎉 NEW OBJECTIVE 2 PIPELINE COMPLETED SUCCESSFULLY!")
        print("Ready for the next phase: Model Training!")
        sys.exit(0)
    else:
        print("\n❌ NEW OBJECTIVE 2 PIPELINE FAILED!")
        print("Please check the logs and fix the issues.")
        sys.exit(1)

if __name__ == "__main__":
    main()
