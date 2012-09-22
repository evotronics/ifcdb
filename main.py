import webapp2

offline_content = """
<html>
  <head>
    <title>IFCDB: Offline</title>
  </head>
  <body>
    <p>IFCDB is currently offline due to an outbreak of raging monkeys!  Check back soon!</p>
  </body>
</html>
"""

class Offline(webapp2.RequestHandler):
    def get(self):
        self.response.set_status(503) # Service Unavailable
        self.response.write(offline_content)

app = webapp2.WSGIApplication([
    ('/.*', Offline),
])
