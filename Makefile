docpush: 
	- git push origin --delete gh-pages
	- git branch -d gh-pages
	git subtree push --prefix docs/build/html origin gh-pages