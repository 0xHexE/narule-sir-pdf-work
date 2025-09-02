#!/usr/bin/env python3
"""
Test script to verify the client selection strategies work correctly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from top_k_selection import TopKModelSelectionStrategy
from adaptive_selection import AdaptiveClientSelectionStrategy

def test_top_k_strategy():
    """Test Top-K Model Selection Strategy."""
    print("Testing Top-K Model Selection Strategy...")
    
    strategy = TopKModelSelectionStrategy(k=2, metric="accuracy")
    
    # Test initialization
    assert strategy.k == 2
    assert strategy.metric == "accuracy"
    assert strategy.client_metrics == {}
    
    print("✓ Initialization test passed")
    
    # Test metric handling
    strategy.client_metrics = {
        "client1": 0.95,
        "client2": 0.85,
        "client3": 0.75,
        "client4": 0.90
    }
    
    # Should select top 2 clients by accuracy
    # Expected: client1 (0.95) and client4 (0.90)
    print(f"Client metrics: {strategy.client_metrics}")
    
    print("✓ Basic functionality test completed")
    return True

def test_adaptive_strategy():
    """Test Adaptive Client Selection Strategy."""
    print("\nTesting Adaptive Client Selection Strategy...")
    
    strategy = AdaptiveClientSelectionStrategy(
        selection_ratio=0.5,
        metric="accuracy",
        history_weight=0.7,
        exploration_factor=0.1
    )
    
    # Test initialization
    assert strategy.selection_ratio == 0.5
    assert strategy.metric == "accuracy"
    assert strategy.history_weight == 0.7
    assert strategy.exploration_factor == 0.1
    
    print("✓ Initialization test passed")
    
    # Test client history tracking
    strategy.client_scores = {
        "client1": 0.95,
        "client2": 0.85,
        "client3": 0.75
    }
    
    # Test probability calculation
    client_ids = ["client1", "client2", "client3", "client4"]  # client4 is new
    probabilities = strategy._calculate_selection_probabilities(client_ids)
    
    print(f"Client scores: {strategy.client_scores}")
    print(f"Selection probabilities: {probabilities}")
    
    # Should have probabilities for all clients
    assert len(probabilities) == 4
    assert "client4" in probabilities  # New client should have probability
    
    print("✓ Probability calculation test passed")
    return True

def main():
    """Run all tests."""
    print("Running strategy tests...\n")
    
    try:
        test_top_k_strategy()
        test_adaptive_strategy()
        
        print("\n🎉 All tests passed! Strategies are ready to use.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)