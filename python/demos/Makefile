step1-hex:
	for res in 0.1 0.09  0.08 0.07 0.06 0.05 0.04 0.03; do \
	  python3 demo_nitsche_unbiased.py --problem=3 --3D --load_steps=1 --outfile="results/single_step_hex.txt" --res=$$res ; \
	done


step1-tet:
	for res in 0.1 0.09  0.08 0.07 0.06 0.05 0.04 0.03; do \
	  python3 demo_nitsche_unbiased.py --problem=3 --3D --load_steps=1 --simplex --outfile="results/single_step_tet.txt" --res=$$res ; \
	done


step5-hex:
	for res in 0.1 0.09  0.08 0.07 0.06 0.05 0.04 0.03; do \
	  python3 demo_nitsche_unbiased.py --problem=3 --3D --load_steps=5 --outfile="results/five_step_hex.txt" --res=$$res ; \
	done


step5-tet:
	for res in 0.1 0.09  0.08 0.07 0.06 0.05 0.04 0.03; do \
	  python3 demo_nitsche_unbiased.py --problem=3 --3D --load_steps=5 --simplex --outfile="results/five_step_tet.txt" --res=$$res ; \
	done