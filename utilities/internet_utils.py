import urllib.request
import json

def get_json_url(url_to_read):
    """
    retrieve a json document from url
    :param url_to_read: url string
    :return :
    """
    
    # get the document
    res = urllib.request.urlopen(url_to_read)
    res_body = res.read()
    
    out = json.loads(res_body.decode("utf-8"))
    return out