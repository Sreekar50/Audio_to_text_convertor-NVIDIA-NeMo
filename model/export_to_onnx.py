from nemo.collections.asr.models import EncDecCTCModel
import nemo.collections.asr as nemo_asr
import os

def export():
    model_path = "stt_hi_conformer_ctc_medium.nemo"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    model = nemo_asr.models.EncDecCTCModel.restore_from(restore_path=model_path)
    model.eval()
    
    # Fix the export method call - use the correct parameter name
    try:
        # Method 1: Try with just output parameter
        model.export(output="stt_hi_conformer_ctc_medium.onnx")
        print("✓ Successfully exported using basic 'output' parameter")
    except Exception as e:
        print(f"Method 1 failed: {e}")
        try:
            # Method 2: Try with check_trace parameter
            model.export(output="stt_hi_conformer_ctc_medium.onnx", check_trace=False)
            print("✓ Successfully exported with check_trace=False")
        except Exception as e2:
            print(f"Method 2 failed: {e2}")
            try:
                # Method 3: Use manual PyTorch ONNX export
                print("Attempting manual PyTorch ONNX export...")
                import torch
                
                # Create dummy inputs matching the model's expected input format
                batch_size = 1
                seq_len = 16000  # 1 second of audio at 16kHz
                
                dummy_audio = torch.randn(batch_size, seq_len)
                dummy_length = torch.tensor([seq_len], dtype=torch.long)
                
                # Manual export using PyTorch
                torch.onnx.export(
                    model,
                    (dummy_audio, dummy_length),
                    "stt_hi_conformer_ctc_medium.onnx",
                    input_names=["audio_signal", "length"],
                    output_names=["logits"],
                    dynamic_axes={
                        "audio_signal": {0: "batch_size", 1: "time"},
                        "length": {0: "batch_size"},
                        "logits": {0: "batch_size", 1: "time"}
                    },
                    opset_version=11,
                    verbose=True
                )
                print("✓ Successfully exported using manual PyTorch ONNX export")
                
            except Exception as e3:
                print(f"All export methods failed: {e3}")
                print("Error details:", str(e3))
                return
    
    try:
        with open("tokens.txt", "w", encoding='utf-8') as f:
            # Handle both vocabulary types
            if hasattr(model.decoder, 'vocabulary'):
                vocab = model.decoder.vocabulary
            elif hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'vocab'):
                vocab = list(model.tokenizer.vocab.keys())
            else:
                print("Warning: Could not find vocabulary")
                vocab = []
            
            for i, s in enumerate(vocab):
                f.write(f"{s} {i}\n")
            f.write(f"<blk> {len(vocab)}\n")
        
        print(f"✓ Tokens file created with {len(vocab)} tokens")
        
    except Exception as e:
        print(f"Error creating tokens file: {e}")
    
    print("Export process completed!")

if __name__ == "__main__":
    export()