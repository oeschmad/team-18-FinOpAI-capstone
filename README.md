<h3 align="center"> AI-Driven Portfolio Optimization for Investors</h3>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#data-access">Data Access</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
### About The Project

Many young, inexperienced investors struggle with structuring their portfolios due to overwhelming 
information and market volatility. Traditional investment strategies, such as 60/40 allocation or 
passive index investing, fail to adapt to changing market conditions and user-specific risk profiles. <br />
This project develops an AI-driven portfolio optimization system leveraging reinforcement learning 
(RL) to dynamically allocate investments across stocks, bonds, cryptocurrencies, and commodities. 
The approach integrates financial market data, economic indicators, and user risk profiles to provide 
personalized investment strategies.

The system aims to: <br />
* Classify investors based on risk tolerance using machine learning. <br />
* Predict asset returns with deep learning models. <br />
* Optimize portfolio allocation using an RL-based model. <br />
* Provide an interactive dashboard for investment insights.<br />


<!-- GETTING STARTED -->
### Getting Started
Follow the steps below to run the project
#### Installation

1. Get a free API Key at [(https://dashboard.ngrok.com/get-started/your-authtoken)]((https://dashboard.ngrok.com/get-started/your-authtoken))
2. Clone the repo
   ```sh
   git clone https://github.com/oeschmad/team-18-FinOpAI-capstone.git
   ```
3. Install required packages from requirements.txt
   ```sh
   pip install -r requirements.txt
   ```
4. Enter your API in `app.py`
   ```js
   NGROK_AUTH_TOKEN = "enter"  # Replace with your token
   ```
5. Run app.py




<!-- USAGE EXAMPLES -->
### Usage
Below is an example of what the dashboard will output for the portfolio allocation:

<img width="1091" alt="Screenshot 2025-04-21 at 2 10 37 PM" src="https://github.com/user-attachments/assets/2fd6b364-db3b-4ce1-ae7f-e628d3a759ca" />

<img width="900" alt="Screenshot 2025-04-21 at 2 11 44 PM" src="https://github.com/user-attachments/assets/8841b430-d7bc-4372-aa77-44315ffaae7f" />



<!-- DATA ACCESS -->
### Data Access
The stock market, precious metals, and crypto data was accessed using the yfinance API, which offers a Pythonic way to fetch financial & market data from Yahoo!Ⓡ finance

The economic and bond data was accessed using the Federal Reserve Economic Data (FRED) API, which is a web service that allows developers to write programs and build applications that retrieve economic data from the FRED® and ALFRED® websites hosted by the Economic Research Division of the Federal Reserve Bank of St. Louis.




<!-- CONTACT -->
### Contact

Team members: 
Madeline Oesch (oeschmad@umich.edu),
Ramya Prakash (ramyapk@umich.edu),
Aniruddh Aithal (aithalan@umich.edu),
Brooke Lee (brojlee@umich.edu)

Project Link: https://github.com/oeschmad/team-18-FinOpAI-capstone




