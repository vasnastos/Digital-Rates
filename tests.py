import requests,json
import requests.auth

def contact_server():
    url = 'http://195.130.73.232/score'
    data = {'key1': 10, 'key2': 20}
    response = requests.post(url, json=data)
    auth=requests.auth.HTTPBasicAuth(username='digitalrates',password='sc0r!ng@pp2024')
    try:
        # Send the POST request
        response = requests.post(url, json=data,auth=auth,verify=False)
        
        # Check if the request was successful
        if response.status_code == 200:
            try:
                # Try to parse the response as JSON
                response_data = response.json()
                print(response_data)
            except json.JSONDecodeError:
                print("Response content is not valid JSON")
                print("Response content:", response.text)
        else:
            print(f"Request failed with status code {response.status_code}")
            print("Response content:", response.text)
    
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

if __name__=='__main__':
    contact_server()