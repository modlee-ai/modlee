# Introduction

### Welcome to Modlee!

Thank you for your interest in contributing to Modlee! Please take a moment to review this document before submitting a pull request.

We continue improving Modlee with feedback from the community.
As we get Modlee off the ground, we welcome contributions of any kind: bug reports, feature requests, tutorials, etc.


### Explain contributions you are NOT looking for (if any).

Again, defining this up front means less work for you. If someone ignores your guide and submits something you don’t want, you can simply close it and point to your policy.

> Please, don't use the issue tracker for [support questions]. Check whether the #pocoo IRC channel on Freenode can help with your issue. If your problem is not strictly Werkzeug or Flask specific, #python is generally more active. Stack Overflow is also worth considering.

[source: [Flask](https://github.com/pallets/flask/blob/master/CONTRIBUTING.rst)] **Need more inspiration?** [1] [cucumber-ruby](https://github.com/cucumber/cucumber-ruby/blob/master/CONTRIBUTING.md#about-to-create-a-new-github-issue) [2] [Read the Docs](http://read-the-docs.readthedocs.org/en/latest/open-source-philosophy.html#unsupported)

# Ground Rules
### Set expectations for behavior (yours, and theirs).
This includes not just how to communicate with others (being respectful, considerate, etc) but also technical responsibilities (importance of testing, project dependencies, etc). Mention and link to your code of conduct, if you have one.

> Responsibilities
> * Ensure cross-platform compatibility for every change that's accepted. Windows, Mac, Debian & Ubuntu Linux.
> * Ensure that code that goes into core meets all requirements in this checklist: https://gist.github.com/audreyr/4feef90445b9680475f2
> * Create issues for any major changes and enhancements that you wish to make. Discuss things transparently and get community feedback.
> * Don't add any classes to the codebase unless absolutely needed. Err on the side of using functions.
> * Keep feature versions as small as possible, preferably one new feature per version.
> * Be welcoming to newcomers and encourage diverse new contributors from all backgrounds. See the [Python Community Code of Conduct](https://www.python.org/psf/codeofconduct/).

[source: [cookiecutter](https://github.com/audreyr/cookiecutter/blob/master/CONTRIBUTING.rst)] **Need more inspiration?** [1] [Celery](https://github.com/celery/celery/blob/master/CONTRIBUTING.rst#community-code-of-conduct) [2] [geocoder](https://github.com/alexreisner/geocoder#contributing)

# Your First Contribution
Help people who are new to your project understand where they can be most helpful. This is also a good time to let people know if you follow a label convention for flagging beginner issues.

> Unsure where to begin contributing to Atom? You can start by looking through these beginner and help-wanted issues:
> Beginner issues - issues which should only require a few lines of code, and a test or two.
> Help wanted issues - issues which should be a bit more involved than beginner issues.
> Both issue lists are sorted by total number of comments. While not perfect, number of comments is a reasonable proxy for impact a given change will have.

[source: [Atom](https://github.com/atom/atom/blob/master/CONTRIBUTING.md#your-first-code-contribution)] **Need more inspiration?** [1] [Read the Docs](http://docs.readthedocs.org/en/latest/contribute.html#contributing-to-development) [2] [Django](https://docs.djangoproject.com/en/dev/internals/contributing/new-contributors/#first-steps) (scroll down to "Guidelines" as well)

### Bonus points: Add a link to a resource for people who have never contributed to open source before.
Here are a couple of friendly tutorials you can include: http://makeapullrequest.com/ and http://www.firsttimersonly.com/

> Working on your first Pull Request? You can learn how from this *free* series, [How to Contribute to an Open Source Project on GitHub](https://egghead.io/series/how-to-contribute-to-an-open-source-project-on-github).

[source: [React](https://github.com/facebook/react/blob/master/CONTRIBUTING.md#pull-requests)]  

As a side note, it helps to use newcomer-friendly language throughout the rest of your document. Here are a couple of examples from [Active Admin](https://github.com/activeadmin/activeadmin/blob/master/CONTRIBUTING.md):

>At this point, you're ready to make your changes! Feel free to ask for help; everyone is a beginner at first :smile_cat:
>
>If a maintainer asks you to "rebase" your PR, they're saying that a lot of code has changed, and that you need to update your branch so it's easier to merge.

# Getting started

### Submitting a contribution

1. Fork this repository.
2. Commit your changes in your fork.
3. Submit a [pull request](https://github.com/modlee-ai/modlee_pypi/pulls).

As a rule of thumb, please bundle small changes (e.g. typographic or grammatic corrections, comments) with larger contributions (e.g. function refactors).

# How to report a bug
### Explain security disclosures first!
If you find a security vulnerability, **please do NOT open an issue**. Please report the vulnerability to [brad@modlee.ai](brad@modlee.ai).

### Tell your contributors how to file a bug report.
You can even include a template so people can just copy-paste (again, less work for you).

When filing a bug report as an issue, please include the following information:
1. Versions of your technical stack:
    - Operating system
    - Python
    - Framework (PyTorch)
2. Actions you took — what steps led to the error?
3. Expected functionality — what did you *expect* to happen?
4. Actual functionality - what *actually* happened?

# How to suggest a feature or enhancement
### If you have a particular roadmap, goals, or philosophy for development, share it here.
This information will give contributors context before they make suggestions that may not align with the project’s needs.

> The Express philosophy is to provide small, robust tooling for HTTP servers, making it a great solution for single page applications, web sites, hybrids, or public HTTP APIs.
>
> Express does not force you to use any specific ORM or template engine. With support for over 14 template engines via Consolidate.js, you can quickly craft your perfect framework.

[source: [Express](https://github.com/expressjs/express#philosophy)] **Need more inspiration?** [Active Admin](https://github.com/activeadmin/activeadmin#goals)

### Explain your desired process for suggesting a feature.
If there is back-and-forth or signoff required, say so. Ask them to scope the feature, thinking through why it’s needed and how it might work.

> If you find yourself wishing for a feature that doesn't exist in Elasticsearch, you are probably not alone. There are bound to be others out there with similar needs. Many of the features that Elasticsearch has today have been added because our users saw the need. Open an issue on our issues list on GitHub which describes the feature you would like to see, why you need it, and how it should work.

[source: [Elasticsearch](https://github.com/elastic/elasticsearch/blob/master/CONTRIBUTING.md#feature-requests)] **Need more inspiration?** [1] [Hoodie](https://github.com/hoodiehq/hoodie/blob/master/CONTRIBUTING.md#feature-requests) [2] [Ember.js](https://github.com/emberjs/ember.js/blob/master/CONTRIBUTING.md#requesting-a-feature)

# Code review process
### Explain how a contribution gets accepted after it’s been submitted.
Who reviews it? Who needs to sign off before it’s accepted? When should a contributor expect to hear from you? How can contributors get commit access, if at all?

> The core team looks at Pull Requests on a regular basis in a weekly triage meeting that we hold in a public Google Hangout. The hangout is announced in the weekly status updates that are sent to the puppet-dev list. Notes are posted to the Puppet Community community-triage repo and include a link to a YouTube recording of the hangout.
> After feedback has been given we expect responses within two weeks. After two weeks we may close the pull request if it isn't showing any activity.

[source: [Puppet](https://github.com/puppetlabs/puppet/blob/master/CONTRIBUTING.md#submitting-changes)] **Need more inspiration?** [1] [Meteor](https://meteor.hackpad.com/Responding-to-GitHub-Issues-SKE2u3tkSiH ) [2] [Express.js](https://github.com/expressjs/express/blob/master/Contributing.md#becoming-a-committer)

# Community
If there are other channels you use besides GitHub to discuss contributions, mention them here. You can also list the author, maintainers, and/or contributors here, or set expectations for response time.

> You can chat with the core team on https://gitter.im/cucumber/cucumber. We try to have office hours on Fridays.

[source: [cucumber-ruby](https://github.com/cucumber/cucumber-ruby/blob/master/CONTRIBUTING.md#talking-with-other-devs)] **Need more inspiration?**
 [1] [Chef](https://github.com/chef/chef/blob/master/CONTRIBUTING.md#-developer-office-hours) [2] [Cookiecutter](https://github.com/audreyr/cookiecutter#community)

# BONUS: Code, commit message and labeling conventions
These sections are not necessary, but can help streamline the contributions you receive.

### Explain your preferred style for code, if you have any.

**Need inspiration?** [1] [Requirejs](http://requirejs.org/docs/contributing.html#codestyle) [2] [Elasticsearch](https://github.com/elastic/elasticsearch/blob/master/CONTRIBUTING.md#contributing-to-the-elasticsearch-codebase)

### Explain if you use any commit message conventions.

**Need inspiration?** [1] [Angular](https://github.com/angular/material/blob/master/.github/CONTRIBUTING.md#submit) [2] [Node.js](https://github.com/nodejs/node/blob/master/CONTRIBUTING.md#step-3-commit)

### Explain if you use any labeling conventions for issues.

**Need inspiration?** [1] [StandardIssueLabels](https://github.com/wagenet/StandardIssueLabels#standardissuelabels) [2] [Atom](https://github.com/atom/atom/blob/master/CONTRIBUTING.md#issue-and-pull-request-labels)