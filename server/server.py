#!/usr/bin/env python

from http.server import BaseHTTPRequestHandler, HTTPServer

# HTTPRequestHandler class
class testHTTPServer_RequestHandler(BaseHTTPRequestHandler):

  # GET
  def do_GET(self):
        # Send response status code
        self.send_response(200)

        # Send headers
        self.send_header('Content-type','text/html')
        self.end_headers()

        # Send message back to client
        message =  "<!DOCTYPE html><html><head><title>Page Title</title></head><body><h1>This is a Heading</h1><p>This is a paragraph.</p></body></html>"
        # Write content as utf-8 data
        self.wfile.write(bytes(message, "utf8"))
        return

  def do_POST(self):
      # Send response status code
      self.send_response(200)

      # Send headers
      self.send_header('Content-type', 'text/html')
      self.end_headers()

      # Send message back to client
      message = "<html><body><h1>POST!</h1></body></html>"
      # Write content as utf-8 data
      self.wfile.write(bytes(message, "utf8"))



def run():
  print('starting server...')

  # Server settings
  # Choose port 8080, for port 80, which is normally used for a http server, you need root access
  server_address = ('127.0.0.1', 3000)
  httpd = HTTPServer(server_address, testHTTPServer_RequestHandler)
  print('running server...')
  httpd.serve_forever()


run()