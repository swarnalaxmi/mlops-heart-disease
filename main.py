#!/usr/bin/env python
"""Main script to orchestrate the complete MLOps pipeline."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='MLOps Heart Disease Pipeline')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Data pipeline
    data_parser = subparsers.add_parser('data', help='Download and process data')
    data_parser.add_argument('--output', default='data/heart_disease.csv', help='Output path')
    
    # Training
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--data', default='data/heart_disease.csv', help='Data path')
    train_parser.add_argument('--output-model', default='models/best_model.pkl', help='Model output path')
    
    # Inference
    infer_parser = subparsers.add_parser('predict', help='Make predictions')
    infer_parser.add_argument('--model', default='models/best_model.pkl', help='Model path')
    infer_parser.add_argument('--preprocessor', default='models/preprocessor.pkl', help='Preprocessor path')
    infer_parser.add_argument('--input', required=True, help='Input JSON file')
    
    # API
    api_parser = subparsers.add_parser('api', help='Start API server')
    api_parser.add_argument('--host', default='0.0.0.0', help='API host')
    api_parser.add_argument('--port', type=int, default=8000, help='API port')
    api_parser.add_argument('--reload', action='store_true', help='Auto-reload on code changes')
    
    # Tests
    test_parser = subparsers.add_parser('test', help='Run tests')
    test_parser.add_argument('--coverage', action='store_true', help='Generate coverage report')
    
    args = parser.parse_args()
    
    if args.command == 'data':
        print("📥 Downloading and processing data...")
        from data_processor import DataProcessor
        processor = DataProcessor()
        df = processor.download_dataset(args.output)
        print(f"✓ Dataset saved to {args.output}")
        
    elif args.command == 'train':
        print("🎯 Training models...")
        from model_trainer import run_training_pipeline
        trainer, processor = run_training_pipeline(args.data, args.output_model)
        print("✓ Training completed!")
        
    elif args.command == 'predict':
        print("🔮 Making predictions...")
        import json
        import numpy as np
        from inference import ModelPredictor
        
        predictor = ModelPredictor(args.model, args.preprocessor)
        
        # Load input
        with open(args.input, 'r') as f:
            input_data = json.load(f)
        
        features = np.array([list(input_data.values())])
        prediction, probability = predictor.predict(features)
        
        print(f"Prediction: {prediction}")
        print(f"Probability: {probability:.4f}")
        
    elif args.command == 'api':
        print(f"🚀 Starting API server on {args.host}:{args.port}...")
        import uvicorn
        uvicorn.run(
            'src.app:app',
            host=args.host,
            port=args.port,
            reload=args.reload
        )
        
    elif args.command == 'test':
        print("✅ Running tests...")
        import subprocess
        cmd = ['pytest', 'tests/', '-v']
        if args.coverage:
            cmd.extend(['--cov=src', '--cov-report=html'])
        subprocess.run(cmd)
        
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
