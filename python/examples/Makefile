all: test

test:
	cp -r ../built-lib/ ./
	python test_gmm.py
	python test_hmm_single.py
	python test_hierarchicalhmm.py

clean:
	rm -r ./xmm/
	rm *results*.txt