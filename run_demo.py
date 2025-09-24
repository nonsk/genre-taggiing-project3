import sys
sys.path.append('src')

def main():
    print("MusicNet: Deep Learning Music Auto-Tagging System")
    print("CNN-RNN Architecture Demo")
    print("=" * 70)
    
    while True:
        print("\nSelect an option:")
        print("1. Validate CV metrics")
        print("2. Compare with baseline model") 
        print("3. Demo music tagging")
        print("4. Evaluate on MagnaTagATune dataset")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            print("\nRunning CV metrics validation...")
            from validate_metrics import validate_cv_metrics
            validate_cv_metrics()
            
        elif choice == '2':
            print("\nComparing model parameters...")
            from src.baseline_comparison import compare_model_parameters
            compare_model_parameters()
            
        elif choice == '3':
            print("\nStarting music tagging demo...")
            from demo_tagging import demonstrate_music_tagging
            demonstrate_music_tagging()
            
        elif choice == '4':
            print("\nEvaluating model performance...")
            from evaluate_performance import evaluate_auc_performance
            evaluate_auc_performance()
            
        elif choice == '5':
            print("Exiting demo...")
            break
            
        else:
            print("Invalid choice. Please select 1-5.")
        
        input("\nPress Enter to continue...")

if __name__ == '__main__':
    main()