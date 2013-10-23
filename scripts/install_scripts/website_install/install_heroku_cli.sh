# per https://devcenter.heroku.com/articles/heroku-command#installing-the-heroku-cli
wget --no-check-certificate -qO- https://toolbelt.heroku.com/install-ubuntu.sh \
	| perl -pe 's/apt-get install/apt-get install --force-yes/' \
	| sh
