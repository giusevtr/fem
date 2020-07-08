ds=adult
python fem_grid_search.py $ds  64  5 0.15
python fem_grid_search.py $ds  64  5 0.2
python fem_grid_search.py $ds  64  5 0.25
python fem_grid_search.py $ds  64  5 0.5
python fem_grid_search.py $ds  64  5 1.0

python fem_grid_search.py $ds  64 3 0.15
python fem_grid_search.py $ds  64 3 0.2
python fem_grid_search.py $ds  64 3 0.25
python fem_grid_search.py $ds  64 3 0.5
python fem_grid_search.py $ds  64 3 1.0

ds=loans
python fem_grid_search.py $ds  64  5 0.15
python fem_grid_search.py $ds  64  5 0.2
python fem_grid_search.py $ds  64  5 0.25
python fem_grid_search.py $ds  64  5 0.5
python fem_grid_search.py $ds  64  5 1.0

python fem_grid_search.py $ds  64 3 0.05
python fem_grid_search.py $ds  64 3 0.15
python fem_grid_search.py $ds  64 3 0.2
python fem_grid_search.py $ds  64 3 0.25
python fem_grid_search.py $ds  64 3 0.5
python fem_grid_search.py $ds  64 3 1.0
