|image0|

.. |image0| image:: https://github.com/mansiagr4/gifs/raw/main/new_small_logo.svg


Grow Your ML Career with Modlee
---------------------
At Modlee, we do more than just provide cutting-edge machine learning tools; we help you grow as an ML developer and land your dream job. You've already explored our package that makes building machine learning models faster and easier. Now, let’s take it a step further—let’s use Modlee to boost your skills, prepare you for real-world challenges, and get you noticed by top AI companies!



Real-World ML Challenges
---------------------
The first step to becoming a standout ML candidate is mastering hands-on experience. Modlee offers a range of real-world exercises that are designed to test and improve your ML capabilities. Whether you’re diving into image classification, time series forecasting, or tabular classification, you'll work with unique datasets to solve real challenges.

Our exercises don’t just prep you for technical interviews—they equip you for the real-world tasks you'll face as an ML engineer. The best part? You can track your progress and instantly compare your performance with others in the Modlee community. Complete your first exercise today and see how you rank after submitting your solution!

.. raw:: html 

    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {
                font-family: Arial, sans-serif;
                display: flex;
                line-height: 1.6;
                margin: 0 auto;
                padding: 20px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
            .colab-link {
                float: right;
                color: #7b33f4;
                text-decoration: none;
            }
            .colab-link:hover {
                text-decoration: underline;
            }
        </style>
    </head>
    <body>
        <h2>Image Classification <a href="https://colab.research.google.com/drive/1GpZe783uDKrdEB6E30LzpUM0QkOW-gr1?usp=sharing#scrollTo=Jy3ZaU0_bkfy" target="_blank" class="colab-link"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a></h2>
        <table id="imageTable">
            <tr>
                <th>Exercise ID</th>
                <th>Model Size Restriction (MB)</th>
            </tr>
            <!-- Image classification exercise IDs will be inserted here -->
        </table>

        <h2>Time Series Forecasting <a href="https://colab.research.google.com/drive/1zxx102WW877wGGTz4MnCSntPVFUr1KZv?usp=sharing#scrollTo=9g_ACcV2_OdF" target="_blank" class="colab-link"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a></h2>
        <table id="timeSeriesTable">
            <tr>
                <th>Exercise ID</th>
                <th>Model Size Restriction (MB)</th>
            </tr>
            <!-- Time series forecasting exercise IDs will be inserted here -->
        </table>

        <h2>Tabular Classification <a href="https://colab.research.google.com/drive/11A0ba7UGcuvF7T5JbQkxXGDI2alqC1k2?usp=sharing" target="_blank" class="colab-link"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a></h2>
        <table id="tabularTable">
            <tr>
                <th>Exercise ID</th>
                <th>Model Size Restriction (MB)</th>
            </tr>
            <!-- Tabular classification exercise IDs will be inserted here -->
        </table>

        <script>
            function fetchExerciseDetails() {
                console.log("HERE")
                fetch("https://evalserver.modlee.ai:5000/docs-real-world-exercises")
                    .then(response => response.json())
                    .then(data => {
                        updateTable('imageTable', data.image);
                        updateTable('timeSeriesTable', data.time_series);
                        updateTable('tabularTable', data.tabular);
                    })
                    .catch(error => console.error('Error:', error));
            }

            function updateTable(tableId, exercises) {
                console.log("NOW HERE")
                const table = document.getElementById(tableId);
                // Clear existing rows
                while (table.rows.length > 1) {
                    table.deleteRow(1);
                }
                exercises.forEach(exercise => {
                    const row = table.insertRow(-1);
                    const cell1 = row.insertCell(0);
                    const cell2 = row.insertCell(1);
                    cell1.textContent = exercise.id;
                    cell2.textContent = exercise.size;
                });
            }

            // Fetch exercise details when the page loads
            window.onload = fetchExerciseDetails();
        </script>
    </body>
    </html>



Join the Talent Pool
-------------------
Ready to put your skills to work? Join our Modlee Talent Pool, where each month we’ll send you assessments to benchmark your skills against peers. We use these results to match top talent with exclusive job opportunities at some of the world’s leading AI companies. If you’re one of the best, we’ll help you get noticed.

.. raw:: html

    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
        body {
                font-family: Arial, sans-serif;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
                background-color: #f0f0f0;
        }
        .signup-form {
                background-color: white;
                padding: 2rem;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 10;
        }
        form {
                display: flex;
                flex-direction: column;
        }
        label {
                margin-bottom: 0.5rem;
        }
        input {
                padding: 0.5rem;
                margin-bottom: 1rem;
                border: 1px solid #ccc;
                border-radius: 4px;
        }
        button {
                padding: 0.5rem 1rem;
                background-color: #7b33f4;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
        }
        button:hover {
                background-color: #995bff;
        }
        #message {
                margin-top: 1rem;
                text-align: center;
                font-weight: bold;
        }
        .empty-container {
            padding-bottom: 100;
        }
        </style>
    </head>
    <body>
        <div class="signup-form">
        <form id="emailForm">
                <label for="email">Email address:</label>
                <input type="email" id="email" name="email" required>
                <button type="submit">Join Modlee Talent Pool</button>
        </form>
        <div id="message"></div>
        </div>

        <div class="empty-container">
            <p></p>
        </div>
        <script>
        document.getElementById('emailForm').addEventListener('submit', function(e) {
                e.preventDefault();
                const email = document.getElementById('email').value;
                const messageElement = document.getElementById('message');

                fetch('https://evalserver.modlee.ai:5000/join_talent_pool', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ email: email }),
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        messageElement.textContent = data.success;
                        messageElement.style.color = 'green';
                        this.reset();
                    } else if (data.warning) {
                        messageElement.textContent = data.warning;
                        messageElement.style.color = 'orange';
                        this.reset();
                    } else if (data.error) {
                        messageElement.textContent = data.error;
                        messageElement.style.color = 'red';
                        this.reset();
                    }
                })
                .catch((error) => {
                    console.error('Error:', error);
                    messageElement.textContent = 'An error occurred. Please try again.';
                    messageElement.style.color = 'red';
                });
        });
        </script>
    </body>
    </html>

Not feeling quite ready to dive into ML interviews yet? No problem! You can keep honing your skills at your own pace with our exercises and join the talent pool whenever you're confident. We're here to support your growth every step of the way.


Curated ML Job Board
---------------
We’ve handpicked a list of current job openings from top AI companies. You can apply directly or, if you’re in our talent pool, we’ll give you a warm introduction to companies hiring for roles that match your skills. This gives you a competitive edge in the job market and helps you stand out from the crowd.


.. raw:: html 

    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0 auto;
                padding: 20px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
            a {
                color: #7b33f4;
                text-decoration: none;
            }
            a:hover {
                text-decoration: underline;
            }
        </style>
    </head>
    <body>
        <h3>Current Job Openings</h3>
        <table id="jobOpeningsTable">
            <thead>
                <tr>
                    <th>Company</th>
                    <th>Role</th>
                    <th>Location</th>
                    <th>Apply</th>
                </tr>
            </thead>
            <tbody>
                <!-- Job openings will be inserted here -->
            </tbody>
        </table>

        <script>
            function fetchJobOpenings() {
                fetch('https://evalserver.modlee.ai:5000/job_board')
                    .then(response => response.json())
                    .then(data => {
                        updateJobOpeningsTable(data);
                    })
                    .catch(error => console.error('Error:', error));
            }

            function updateJobOpeningsTable(jobOpenings) {
                const table = document.getElementById('jobOpeningsTable');
                const tbody = table.querySelector('tbody');

                // Clear existing rows
                tbody.innerHTML = '';

                jobOpenings.forEach(job => {
                    const row = tbody.insertRow();
                    row.innerHTML = `
                        <td>${job.company_name}</td>
                        <td>${job.title}</td>
                        <td>${job.location}</td>
                        <td><a href="${job.apply_link}" target="_blank">Link</a></td>
                    `;
                });
            }

            // Fetch job openings when the page loads
            window.onload = fetchJobOpenings;
        </script>
    </body>
    </html>



Next Steps
-----------
Here’s how to get started:

- Test Your Skills: Work through our real-world ML challenges at your own pace.
- Join the Modlee Talent Pool: Benchmark your skills against the Modlee Community.
- Get Noticed: If you’re at the top of our talent pool, we’ll connect you with leading AI companies and help you land your next big opportunity.

Your journey to becoming a top ML developer starts here. Ready to take the next step? Start honing your skills with Modlee today!



---------------

.. |image0| image:: https://github.com/mansiagr4/gifs/raw/main/new_small_logo.svg