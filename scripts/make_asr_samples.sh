#!/usr/bin/env bash
# Synthesize the ASR eval audio set with macOS `say` voices (en_IN/hi_IN/te_IN).
# Usage: bash scripts/make_asr_samples.sh
set -euo pipefail
mkdir -p data/benchmark/audio

say -v Rishi -o data/benchmark/audio/en1.aiff "I am a farmer with two acres of land. Can I get PM-KISAN benefits?"
say -v Rishi -o data/benchmark/audio/en2.aiff "My father earns two lakh rupees per year. Am I eligible for the post matric scholarship?"
say -v Lekha -o data/benchmark/audio/hi1.aiff "मैं एक किसान हूँ। क्या मुझे पीएम-किसान योजना का लाभ मिलेगा?"
say -v Lekha -o data/benchmark/audio/hi2.aiff "मेरे पिता की सालाना आय दो लाख रुपये है। क्या मुझे छात्रवृत्ति मिलेगी?"
say -v Geeta -o data/benchmark/audio/te1.aiff "నేను రైతుని. నాకు పీఎం-కిసాన్ పథకం కింద డబ్బు వస్తుందా?"
say -v Geeta -o data/benchmark/audio/te2.aiff "నేను ఎస్సీ విద్యార్థిని. నాకు పోస్ట్ మెట్రిక్ స్కాలర్‌షిప్ వస్తుందా?"

cat > data/benchmark/asr_eval.jsonl <<'EOF'
{"audio":"data/benchmark/audio/en1.aiff","reference":"I am a farmer with two acres of land. Can I get PM-KISAN benefits?","lang":"en"}
{"audio":"data/benchmark/audio/en2.aiff","reference":"My father earns two lakh rupees per year. Am I eligible for the post matric scholarship?","lang":"en"}
{"audio":"data/benchmark/audio/hi1.aiff","reference":"मैं एक किसान हूँ। क्या मुझे पीएम-किसान योजना का लाभ मिलेगा?","lang":"hi"}
{"audio":"data/benchmark/audio/hi2.aiff","reference":"मेरे पिता की सालाना आय दो लाख रुपये है। क्या मुझे छात्रवृत्ति मिलेगी?","lang":"hi"}
{"audio":"data/benchmark/audio/te1.aiff","reference":"నేను రైతుని. నాకు పీఎం-కిసాన్ పథకం కింద డబ్బు వస్తుందా?","lang":"te"}
{"audio":"data/benchmark/audio/te2.aiff","reference":"నేను ఎస్సీ విద్యార్థిని. నాకు పోస్ట్ మెట్రిక్ స్కాలర్‌షిప్ వస్తుందా?","lang":"te"}
EOF
echo "wrote $(ls data/benchmark/audio | wc -l) audio files + manifest"
