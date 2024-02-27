gitpush: 
	- git push -d origin gh-pages
	- git branch -D gh-pages
	git subtree push --prefix docs/build/html origin gh-pages