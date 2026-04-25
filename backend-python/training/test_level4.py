import torch
import time

from level4_temporal_gate import TemporalContextLSTM, final_decision_gate

def run_tests():
    print("=" * 60)
    print(" AuraOS: Verification of Stage 3 Temporal LSTM & Decision Gate ")
    print("=" * 60)
    print("Improvements: Self-attention over LSTM outputs, deeper classifier")
    
    # Initialize the Temporal Context Bi-LSTM
    print("\nLoading Temporal Bi-LSTM + Attention (Stage 3)...")
    lstm_model = TemporalContextLSTM()
    lstm_model.eval()
    
    # Create test input arrays mimicking 30 seconds of rolling history
    # Tensor shape is [Batch=1, SequenceLength=6]
    
    # 1. Panic Escalation: Arousal steadily rises towards critical over 30s
    escalating_arousal = torch.tensor([[4.0, 5.2, 6.5, 7.8, 8.5, 9.1]], dtype=torch.float32)
    
    # 2. Happy Cheering: Arousal is high but erratic/bouncy, not organically escalating
    erratic_cheering = torch.tensor([[4.0, 9.0, 4.5, 8.0, 3.5, 7.6]], dtype=torch.float32)
    
    # Benchmark Inference
    print("\nBenchmarking Bi-LSTM + Attention Forward Pass latency...")
    
    # Warm-up
    with torch.no_grad():
        _ = lstm_model(escalating_arousal)
    
    start = time.perf_counter()
    with torch.no_grad():
        score = lstm_model(escalating_arousal)
    dt = (time.perf_counter() - start) * 1000
    print(f" Execution time: {dt:.2f} ms")
    if dt < 40.0:
        print(" [PASS] Execution completes within the < 40ms constraint!")
    else:
        print(" [WARN] Execution missed the 40ms constraint.")
        
    print("\n" + "-" * 60)
    print(" Running Triage Simulations ")
    print("-" * 60)

    # Note: Because the Bi-LSTM has randomly initialized weights currently, 
    # the exact escalation probability is randomized. For the test of the 
    # strict decision logic, we will mock the LSTM string predictions.
    
    # Case A: True Panic Escalation
    # Physiology confirms tremors. Arousal is currently 8.5. 
    # LSTM confirms organic escalation.
    triage_A = final_decision_gate(
        stage1_physiology=True, 
        stage2_arousal_score=8.5, 
        stage3_escalation_prob=0.85 # LSTM says YES it's escalating
    )
    print("Scenario A: True Panic Escalation")
    print(f"Triggered? {triage_A['alert_triggered']}   (Expected: True)\n")
    
    # Case B: Happy Cheering (False Positive Prevention)
    # Friend tells a joke loudly. Arousal is 8.0. 
    # BUT, physiology has no vocal tremor (False) and no steady escalation build-up.
    triage_B = final_decision_gate(
        stage1_physiology=False, 
        stage2_arousal_score=8.0, 
        stage3_escalation_prob=0.20 # LSTM says NO escalation
    )
    print("Scenario B: Happy Cheering (No Tremor)")
    print(f"Triggered? {triage_B['alert_triggered']}  (Expected: False)\n")
    
    # Case C: Throat Clearing / Short Bump
    # Microphone gets hit. Noise injects false physical tremor (True). 
    # Arousal is 5.0 (Low). No escalation prior.
    triage_C = final_decision_gate(
        stage1_physiology=True, 
        stage2_arousal_score=5.0, 
        stage3_escalation_prob=0.10 # LSTM says NO escalation
    )
    print("Scenario C: Short Bump/Cough")
    print(f"Triggered? {triage_C['alert_triggered']}  (Expected: False)\n")
    
    # Case D: Severe Instant Panic (Extreme Spike bypasses LSTM)
    # User screams in pure terror. Physiology confirms severe tremor. Score is 9.5!
    # Even if LSTM hasn't seen 30 seconds of build-up (escapes temporal pattern), it must fire.
    triage_D = final_decision_gate(
        stage1_physiology=True, 
        stage2_arousal_score=9.5, 
        stage3_escalation_prob=0.10 # LSTM didn't detect pattern yet
    )
    print("Scenario D: Severe Instant Panic")
    print(f"Triggered? {triage_D['alert_triggered']}   (Expected: True)\n")
    
    if triage_A['alert_triggered'] and not triage_B['alert_triggered'] and not triage_C['alert_triggered'] and triage_D['alert_triggered']:
        print("=" * 60)
        print(" ALL LOGIC GATES VERIFIED PERFECTLY! ")
        print("=" * 60)

if __name__ == "__main__":
    run_tests()
