python scripts/inference.py --exp_dir=experiment --checkpoint_path=pretrained_models/hyperstyle_ffhq.pt  --data_path=test_data/aligned  --test_batch_size=1 --test_workers=0 --n_iters_per_batch=5 --load_w_encoder --w_encoder_checkpoint_path pretrained_models/faces_w_encoder.pt
python scripts/align_faces_parallel.py --num_threads=1 --root_path=test_data
python scripts/run_domain_adaptation.py --exp_dir=experiment --checkpoint_path=pretrained_models/hyperstyle_ffhq.pt --finetuned_generator_checkpoint_path=pretrained_models/nada/pixar.pt --data_path=test_data/aligned  --test_batch_size=1 --test_workers=0 --n_iters_per_batch=5 --load_w_encoder --w_encoder_checkpoint_path pretrained_models/faces_w_encoder.pt --restyle_n_iterations=2










